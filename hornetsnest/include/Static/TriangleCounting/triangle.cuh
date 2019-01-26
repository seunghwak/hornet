#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using triangle_t = unsigned int;

struct TriangleData {
    TriangleData(const gpu::Hornet<EMPTY,EMPTY>& hornet){
        nv = hornet.nV();
        ne = hornet.nE();
        triPerVertex=NULL;
    }

    int tsp;
    int nbl;
    int shifter;
    int blocks;
    int sps;

    int threadBlocks;
    int blockSize;
    int threadsPerIntersection;
    int logThreadsPerInter;
    int numberInterPerBlock;

    triangle_t* triPerVertex;

    vid_t nv;
    off_t ne;           // undirected-edges
};

//==============================================================================

// Label propogation is based on the values from the previous iteration.
class TriangleCounting : public StaticAlgorithm<gpu::Hornet<EMPTY,EMPTY>> {
public:
    TriangleCounting(gpu::Hornet<EMPTY,EMPTY>& hornet);
    ~TriangleCounting();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override { return true; }

    void run(int cutoff);

    void init();
    void setInitParameters(int threadBlocks, int blockSize, int threadsPerIntersection);
    triangle_t countTriangles();

private:
    bool memReleased;
    HostDeviceVar<TriangleData> hd_triangleData;
};

//==============================================================================

} // namespace hornets_nest
