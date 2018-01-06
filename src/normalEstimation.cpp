//
// Created by Aron Monszpart on 19/01/17.
//

#include "acq/normalEstimation.h"

#include "acq/impl/normalEstimation.hpp" // Templated functions

#include "nanoflann/nanoflann.hpp"  // Nearest neighbour lookup in a pointcloud

#include <queue>
#include <set>
#include <iostream>

namespace acq {

NeighboursT
calculateCloudNeighbours(
    CloudT  const& cloud,
    int     const  k,
    float   const  maxDist,
    int     const  maxLeafs
) {
    // Floating point type
    typedef typename CloudT::Scalar Scalar;
    // Point dimensions
    enum { Dim = 3 };
    // Copy-free Eigen->FLANN wrapper
    typedef nanoflann::KDTreeEigenMatrixAdaptor <
        /*    Eigen matrix type: */ CloudT,
        /* Space dimensionality: */ Dim,
        /*      Distance metric: */ nanoflann::metric_L2
    > KdTreeWrapperT;

    // Squared max distance
    float const maxDistSqr = maxDist * maxDist;

    // Safety check dimensionality
    if (cloud.cols() != Dim) {
        std::cerr << "Point dimension mismatch: " << cloud.cols()
                  << " vs. " << Dim
                  << "\n";
        throw new std::runtime_error("Point dimension mismatch");
    } //...check dimensionality

    // Build KdTree
    KdTreeWrapperT cloudIndex(Dim, cloud, maxLeafs);
    cloudIndex.index->buildIndex();

    // Neighbour indices
    std::vector<size_t> neighbourIndices(k);
    std::vector<Scalar> distsSqr(k);

    // Placeholder structure for nanoFLANN
    nanoflann::KNNResultSet <Scalar> resultSet(k);

    // Associative list of neighbours: { pointId => [neighbourId_0, nId_1, ... nId_k-1] }
    NeighboursT neighbours;
    // For each point, store normal
    for (int pointId = 0; pointId != cloud.rows(); ++pointId) {
        // Initialize nearest neighhbour estimation
        resultSet.init(&neighbourIndices[0], &distsSqr[0]);

        // Make sure it's ok to expose raw data pointer of point
        static_assert(
            std::is_same<Scalar, double>::value,
            "Double input assumed next. Otherwise, explicit copy is needed!"
        );

        // Find neighbours of point in "pointId"-th row
        cloudIndex.index->findNeighbors(
            /*                Output wrapper: */ resultSet,
            /* Query point double[3] pointer: */ cloud.row(pointId).data(),
            /*    How many neighbours to use: */ nanoflann::SearchParams(k)
        );

        // Filter neighbours by squared distance
        NeighboursT::mapped_type currNeighbours;
        for (int i = 0; i != neighbourIndices.size(); ++i) {
            // if not same point and close enough
            if ((neighbourIndices[i] != pointId   ) &&
                (distsSqr        [i] <  maxDistSqr))
                currNeighbours.insert(neighbourIndices[i]);
        }

        // Store list of neighbours
        std::pair<NeighboursT::iterator, bool> const success =
            neighbours.insert(
                std::make_pair(
                    pointId,
                    currNeighbours
                )
            );

        if (!success.second)
            std::cerr << "Could not store neighbours of point " << pointId
                      << ", already inserted?\n";

    } //...for all points

    // return estimated normals
    return neighbours;
} //...calculateCloudNormals()

NormalsT
calculateCloudNormals(
    CloudT      const& cloud,
    NeighboursT const& neighbours
) {
    // Output normals: N x 3
    CloudT normals(cloud.rows(), 3);

    // For each point, store normal
    for (int pointId = 0; pointId != cloud.rows(); ++pointId) {
        // Estimate vertex normal from neighbourhood indices and cloud
        normals.row(pointId) =
            calculatePointNormal(
                /*        PointCloud: */ cloud,
                /*      ID of vertex: */ pointId,
                /* Ids of neighbours: */ neighbours.at(pointId)
            );
    } //...for all points

    // Return estimated normals
    return normals;
} //...calculateCloudNormals()

int
orientCloudNormals(
    NeighboursT const& neighbours,
    NormalsT         & normals
) {
    if (!normals.size()) {
        std::cerr << "[orientCloudNormals] No normals to work on...\n";
        return -1;
    }

    // List of points to visit
    std::queue<int> queue;
    // Quick-lookup unique container of already visited points
    std::set<int> visited;

    // Count changes
    int nFlips = 0;

    while (visited.size() != normals.size()) {
        // Traverse a connected component
        if (queue.empty()) {
            if (!visited.size()) {
                // Initialize queue with one random point
                queue.push(rand() % normals.size()); // TODO: pick point with low curvature
            } else {
                // Expand queue with first unvisited point
                for (int i = 0; i != normals.size() && queue.empty(); ++i) {
                    // if unvisited, use
                    if (visited.find(i) == visited.end())
                        queue.push(i); // enqueue
                } //...for each point
            } //...next component

            // Set visited
            visited.insert(queue.front());
        } //...if queue empty

        // While points to visit exist
        while (!queue.empty()) {
            // Read next point from queue
            int const pointId = queue.front();
            // Remove point from queue
            queue.pop();

            // Fetch neighbours
            NeighboursT::const_iterator const iter = neighbours.find(pointId);
            // Check, if any neighbours
            if (iter == neighbours.end()) {
                //std::cerr << "Could not find neighbours of point " << pointId << "\n";
                continue;
            }

            for (int const neighbourId : iter->second) {
                // If unvisited
                if (visited.find(neighbourId) == visited.end()) {
                    // Enqueue for next level
                    queue.push(neighbourId);
                    // Mark visited
                    visited.insert(neighbourId);

                    // Flip neighbour normal, if not same direction as precursor point
                    if (normals.row(pointId).dot(normals.row(neighbourId)) < 0.f) {
                        normals.row(neighbourId) *= -1.f;
                        ++nFlips;
                    }
                } //...if neighbour unvisited
            } //...for each neighbour of point
        } //...while points in queue
    }

    return nFlips;
} //...orientCloudNormals()

} //...ns acq


//
// Template instantiation
//

namespace acq {

template int
orientCloudNormalsFromFaces(
    FacesT  const& faces,
    NormalsT     & normals
);

template NeighboursT
calculateCloudNeighboursFromFaces(
    FacesT const& faces
);

} //...ns acq
