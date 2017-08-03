

// #include "update.hpp"
// #include "cuStinger.hpp"
// #include "operators.cuh"
// #include "static_katz_centrality/katz.cuh"

#include "Static/KatzCentrality/katz.cuh"


// #include "kernel_mergesort.hxx"
// using namespace mgpu;

typedef int32_t length_t;


namespace custinger_alg {

/// TODO - changed hostKatzdata to pointer so that I can try to inherit it in the streaming case.

katzCentrality::katzCentrality(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
                                       load_balacing(custinger)
									    {

    deviceKatzData = register_data(hostKatzData);
	memReleased = true;
}

katzCentrality::~katzCentrality() {
	release();
    // gpu::free(hostKatzData.distances);
}	


// void katzCentrality::setInitParameters(length_t maxIteration_, length_t K_,length_t maxDegree_,bool isStatic_){
void katzCentrality::setInitParameters(int32_t maxIteration_,int32_t K_,int32_t maxDegree_, bool isStatic_){
	hostKatzData.K=K_;
	hostKatzData.maxDegree=maxDegree_;
	hostKatzData.alpha = 1.0/((double)hostKatzData.maxDegree+1.0);

	hostKatzData.maxIteration=maxIteration_;
	isStatic = isStatic_;

	if(maxIteration_==0){
		cout << "Number of max iterations should be greater than zero" << endl;
		return;
	}
}


void katzCentrality::init(cuStinger& custing){

	if(memReleased==false){
		release();
		memReleased=true;
	}
	hostKatzData.nv = custing.nV();

	if(isStatic==true){
		gpu::allocate(hostKatzData.nPathsData, hostKatzData.nv*2);
		// hostKatzData.nPathsData = (ulong_t*) allocDeviceArray(2*(hostKatzData.nv), sizeof(ulong_t));
		hostKatzData.nPathsPrev = hostKatzData.nPathsData;
		hostKatzData.nPathsCurr = hostKatzData.nPathsData+(hostKatzData.nv);
	}
	else{
		gpu::allocate(hostKatzData.nPathsData, hostKatzData.nv*hostKatzData.maxIteration);
		gpu::allocate(hostKatzData.nPaths, hostKatzData.maxIteration);
		// hostKatzData.nPathsData = (ulong_t*) allocDeviceArray((hostKatzData.nv)*hostKatzData.maxIteration, sizeof(ulong_t));
		// hostKatzData.nPaths = (ulong_t**) allocDeviceArray(hostKatzData.maxIteration, sizeof(ulong_t*));

		// hPathsPtr = (ulong_t**)allocHostArray(hostKatzData.maxIteration, sizeof(ulong_t*));
		hPathsPtr = (ulong_t**)malloc(hostKatzData.maxIteration* sizeof(ulong_t*));

		for(int i=0; i< hostKatzData.maxIteration; i++){
			hPathsPtr[i] = (hostKatzData.nPathsData+(hostKatzData.nv)*i);
		}
		hostKatzData.nPathsPrev = hPathsPtr[0];
		hostKatzData.nPathsCurr = hPathsPtr[1];

		// copyArrayHostToDevice(hPathsPtr,hostKatzData.nPaths,hostKatzData.maxIteration,sizeof(double));
		copyHostToDevice(hPathsPtr,hostKatzData.nPaths,hostKatzData.maxIteration);
	}

	gpu::allocate(hostKatzData.KC, hostKatzData.nv);
	gpu::allocate(hostKatzData.lowerBound, hostKatzData.nv);
	gpu::allocate(hostKatzData.upperBound, hostKatzData.nv);
	// hostKatzData.KC         = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));
	// hostKatzData.lowerBound = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));
	// hostKatzData.upperBound = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));

	gpu::allocate(hostKatzData.isActive, hostKatzData.nv);
	gpu::allocate(hostKatzData.indexArray, hostKatzData.nv);
	gpu::allocate(hostKatzData.vertexArray, hostKatzData.nv);
	gpu::allocate(hostKatzData.lowerBoundSort, hostKatzData.nv);
	// hostKatzData.isActive 		 = (bool*) allocDeviceArray(hostKatzData.nv, sizeof(bool));
	// hostKatzData.indexArray	 =  (vid_t*) allocDeviceArray(hostKatzData.nv, sizeof(vid_t));
	// hostKatzData.vertexArray	 =  (vid_t*) allocDeviceArray(hostKatzData.nv, sizeof(vid_t));
	// hostKatzData.lowerBoundSort = (double*) allocDeviceArray(hostKatzData.nv, sizeof(double));

	// deviceKatzData = (katzData*)allocDeviceArray(1, sizeof(katzData));
	// cusLB = new cusLoadBalance(custing);

	syncDeviceWithHost();
	reset();
}

void katzCentrality::reset(){
	hostKatzData.iteration = 1;

	if(isStatic==true){
		hostKatzData.nPathsPrev = hostKatzData.nPathsData;
		hostKatzData.nPathsCurr = hostKatzData.nPathsData+(hostKatzData.nv);
	}
	else{
		hostKatzData.nPathsPrev = hPathsPtr[0];
		hostKatzData.nPathsCurr = hPathsPtr[1];
	}
	syncDeviceWithHost();
}

void katzCentrality::release(){
	if(memReleased==true)
		return;
	memReleased=true;

	// delete cusLB;
	gpu::free(hostKatzData.nPathsData);

	if (!isStatic){
		gpu::free(hostKatzData.nPaths);
		// freeHostArray(hPathsPtr);
		free(hPathsPtr);
	}

	gpu::free(hostKatzData.vertexArray);
	gpu::free(hostKatzData.KC);
	gpu::free(hostKatzData.lowerBound);
	gpu::free(hostKatzData.lowerBoundSort);
	gpu::free(hostKatzData.upperBound);

	gpu::free(deviceKatzData);
}

// void katzCentrality::run(cuStinger& custing){
void katzCentrality::run(){
	// allVinG_TraverseVertices<katzCentralityOperator::init>(custing,deviceKatzData);
	forAllVertices<katz_operators::init>(custinger,deviceKatzData);
	standard_context_t context(false);

	hostKatzData.iteration = 1;
	
	hostKatzData.nActive = hostKatzData.nv;
	while(hostKatzData.nActive> hostKatzData.K && hostKatzData.iteration < hostKatzData.maxIteration){

		hostKatzData.alphaI          = pow(hostKatzData.alpha,hostKatzData.iteration);
		hostKatzData.lowerBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha));
		hostKatzData.upperBoundConst = pow(hostKatzData.alpha,hostKatzData.iteration+1)/((1.0-hostKatzData.alpha*(double)hostKatzData.maxDegree));
		hostKatzData.nActive = 0; // Each iteration the number of active vertices is set to zero.
	
		syncDeviceWithHost(); // Passing constants to the device.

		// allVinG_TraverseVertices<katzCentralityOperator::initNumPathsPerIteration>(custing,deviceKatzData);
		// allVinA_TraverseEdges_LB<katzCentralityOperator::updatePathCount>(custing,deviceKatzData,*cusLB);
		// allVinG_TraverseVertices<katzCentralityOperator::updateKatzAndBounds>(custing,deviceKatzData);

		forAllVertices<katz_operators::initNumPathsPerIteration>(custinger,deviceKatzData);
        forAllEdges<katz_operators::updatePathCount>(custinger, deviceKatzData);
		// allVinA_TraverseEdges_LB<katzCentralityOperator::updatePathCount>(custing,deviceKatzData,*cusLB);
		// allVinG_TraverseVertices<katzCentralityOperator::updateKatzAndBounds>(custing,deviceKatzData);



		syncHostWithDevice();
		hostKatzData.iteration++;

		if(isStatic){
			// Swapping pointers.
			ulong_t* temp = hostKatzData.nPathsCurr; hostKatzData.nPathsCurr=hostKatzData.nPathsPrev; hostKatzData.nPathsPrev=temp;	
		}else{
			hostKatzData.nPathsPrev = hPathsPtr[hostKatzData.iteration - 1];
			hostKatzData.nPathsCurr = hPathsPtr[hostKatzData.iteration - 0];
		}

		length_t oldActiveCount = hostKatzData.nActive;
		hostKatzData.nActive = 0; // Resetting active vertices for sorting operations.

		syncDeviceWithHost();

		mergesort(hostKatzData.lowerBoundSort,hostKatzData.vertexArray,oldActiveCount, greater_t<double>(),context);

		// allVinG_TraverseVertices<katzCentralityOperator::countActive>(custing,deviceKatzData);
		allVinA_TraverseVertices<katzCentralityOperator::countActive>(custing,deviceKatzData,hostKatzData.vertexArray,oldActiveCount);


// /* 	ulong_t* nPathsCurr = (ulong_t*) allocHostArray(hostKatzData.nv, sizeof(ulong_t));
// 	ulong_t* nPathsPrev = (ulong_t*) allocHostArray(hostKatzData.nv, sizeof(ulong_t));
// 	vid_t* vertexArray = (vid_t*) allocHostArray(hostKatzData.nv, sizeof(vid_t));
// 	double* KC         = (double*) allocHostArray(hostKatzData.nv, sizeof(double));
// 	double* lowerBound = (double*) allocHostArray(hostKatzData.nv, sizeof(double));
// 	double* upperBound = (double*) allocHostArray(hostKatzData.nv, sizeof(double));
    
// 	copyArrayDeviceToHost(hostKatzData.lowerBound,lowerBound,custing.nv, sizeof(double)) ;
// 	copyArrayDeviceToHost(hostKatzData.upperBound,upperBound,custing.nv, sizeof(double)) ;
// 	copyArrayDeviceToHost(hostKatzData.KC,KC,custing.nv, sizeof(double)) ;
// 	copyArrayDeviceToHost(hostKatzData.vertexArray,vertexArray,custing.nv, sizeof(vid_t)) ;

// //	for (int i=0; i<10; i++){
// //	  printf("%d : katz = %g    lower = %g    upper=%g\n",vertexArray[i], KC[vertexArray[i]],lowerBound[vertexArray[i]],upperBound[vertexArray[i]]);
// //	}

//   	freeHostArray(nPathsCurr);
// 	freeHostArray(nPathsPrev);
//     freeHostArray(vertexArray);
// 	freeHostArray(KC);
//     freeHostArray(lowerBound);
// 	freeHostArray(upperBound);
// */		
		syncHostWithDevice();
		cout << hostKatzData.nActive << endl;
	}
	// cout << "@@ " << hostKatzData.iteration << " @@" << endl;
	syncHostWithDevice();
}

length_t katzCentrality::getIterationCount(){
	syncHostWithDevice();
	return hostKatzData.iteration;
}


}// cuStingerAlgs namespace
