#pragma once

#include "HornetAlg.hpp"

namespace hornets_nest {

using triangle_t = unsigned int;
//using triangle_t = unsigned long long;

class TriangleCounting : public StaticAlgorithm<gpu::Hornet<EMPTY, EMPTY>> {
public:
    TriangleCounting(gpu::Hornet<EMPTY, EMPTY>& hornet);
    ~TriangleCounting();

    void reset() override;
    void run() override;
    void release() override;
    bool validate() override { return true; }

    void run(const int WORK_FACTOR);
    void init();
    void copyTCToHost(triangle_t* h_tcs);

    triangle_t countTriangles();

protected:
   triangle_t* triPerVertex { nullptr };

};

}

