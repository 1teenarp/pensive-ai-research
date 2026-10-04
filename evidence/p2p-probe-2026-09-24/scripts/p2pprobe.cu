#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("    CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); fflush(stdout); exit(2);} }while(0)

// GPU0 writes a pattern directly into GPU1 memory (SM-initiated peer store)
__global__ void k_peer_store(volatile int* peer, int n, int val){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) peer[i] = val + i;
}
// GPU0 does atomics on GPU1 memory
__global__ void k_peer_atomic(int* peer, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) atomicAdd(&peer[0], 1);
}
// NCCL's exact pattern: producer peer-writes payload, fences, then sets flag
__global__ void k_producer(volatile int* peer_data, volatile int* peer_flag, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) peer_data[i] = 0xABC0000 + i;
    __threadfence_system();
    if(i==0) *peer_flag = 1;
}
// consumer spins on its OWN local memory (bounded), reports what it saw
__global__ void k_consumer(volatile int* flag, volatile int* data, int* out, long long maxspin){
    if(threadIdx.x==0 && blockIdx.x==0){
        long long s=0;
        while(*flag==0 && s<maxspin) s++;
        out[0] = *flag;      // did flag arrive?
        out[1] = data[0];    // did payload arrive?
        out[2] = data[1023];
        out[3] = (int)(s<maxspin);
    }
}

int main(int argc,char**argv){
    int test = atoi(argv[1]);
    int N=1024*1024;
    int canA=0,canB=0;
    CK(cudaDeviceCanAccessPeer(&canA,0,1));
    CK(cudaDeviceCanAccessPeer(&canB,1,0));
    printf("    canAccessPeer 0->1=%d 1->0=%d\n",canA,canB); fflush(stdout);
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1,0));
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0,0));

    int *d1=nullptr;                      // buffer on GPU1
    CK(cudaSetDevice(1)); CK(cudaMalloc(&d1,N*sizeof(int))); CK(cudaMemset(d1,0,N*sizeof(int)));
    int* h=(int*)malloc(N*sizeof(int));

    if(test==1){ // ---- SM-initiated peer STORE ----
        CK(cudaSetDevice(0));
        k_peer_store<<<(N+255)/256,256>>>((volatile int*)d1,N,0x5000000);
        CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(1)); CK(cudaMemcpy(h,d1,N*sizeof(int),cudaMemcpyDeviceToHost));
        long bad=0; for(int i=0;i<N;i++) if(h[i]!=0x5000000+i) bad++;
        printf("    peer STORE: mismatches %ld / %d  -> %s\n",bad,N, bad==0?"WORKS":"FAILED");
    }
    else if(test==2){ // ---- SM-initiated peer ATOMIC ----
        CK(cudaSetDevice(0));
        k_peer_atomic<<<64,256>>>(d1,64*256);
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){ printf("    peer ATOMIC: launch/sync error: %s -> FAILED\n",cudaGetErrorString(e)); return 0; }
        CK(cudaSetDevice(1)); CK(cudaMemcpy(h,d1,sizeof(int),cudaMemcpyDeviceToHost));
        printf("    peer ATOMIC: counter=%d expected=%d -> %s\n",h[0],64*256, h[0]==64*256?"WORKS":"FAILED");
    }
    else if(test==3){ // ---- NCCL pattern: peer write + fence + flag, consumer spins locally ----
        int *flag=nullptr,*out=nullptr;
        CK(cudaSetDevice(1));
        CK(cudaMalloc(&flag,sizeof(int)));  CK(cudaMemset(flag,0,sizeof(int)));
        CK(cudaMalloc(&out,4*sizeof(int))); CK(cudaMemset(out,0,4*sizeof(int)));
        cudaStream_t sc; CK(cudaStreamCreate(&sc));
        k_consumer<<<1,1,0,sc>>>((volatile int*)flag,(volatile int*)d1,out,3000000000LL);
        CK(cudaSetDevice(0));
        k_producer<<<(N+255)/256,256>>>((volatile int*)d1,(volatile int*)flag,N);
        CK(cudaDeviceSynchronize());
        CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(sc));
        int o[4]; CK(cudaMemcpy(o,out,4*sizeof(int),cudaMemcpyDeviceToHost));
        printf("    flag_seen=%d payload[0]=0x%X payload[1023]=0x%X spin_exited_early=%d\n",o[0],o[1],o[2],o[3]);
        printf("    -> flag visibility: %s | payload visibility: %s\n",
               o[0]==1?"WORKS":"FAILED", (o[1]==0xABC0000)?"WORKS":"FAILED");
    }
    else if(test==4){ // ---- control: copy-engine peer memcpy ----
        int* d0=nullptr; CK(cudaSetDevice(0)); CK(cudaMalloc(&d0,N*sizeof(int)));
        for(int i=0;i<N;i++) h[i]=0x7000000+i;
        CK(cudaMemcpy(d0,h,N*sizeof(int),cudaMemcpyHostToDevice));
        CK(cudaMemcpyPeer(d1,1,d0,0,N*sizeof(int)));
        CK(cudaDeviceSynchronize());
        memset(h,0,N*sizeof(int));
        CK(cudaSetDevice(1)); CK(cudaMemcpy(h,d1,N*sizeof(int),cudaMemcpyDeviceToHost));
        long bad=0; for(int i=0;i<N;i++) if(h[i]!=0x7000000+i) bad++;
        printf("    memcpyPeer (copy engine): mismatches %ld / %d -> %s\n",bad,N,bad==0?"WORKS":"FAILED");
    }
    return 0;
}
