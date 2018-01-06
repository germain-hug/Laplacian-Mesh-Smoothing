//
// Created by bontius on 20/01/17.
//

#include "acq/impl/cloudManager.hpp"
#include <iostream>


namespace acq {

void CloudManager::addCloud(DecoratedCloud const& cloud) {
    _clouds.push_back(cloud);
} //...CloudManager::addCloud()

void CloudManager::setCloud(DecoratedCloud const& cloud, int index) {
    if (index >= _clouds.size()) {
        if (index != _clouds.size())
            std::cerr << "[CloudManager::setCloud] "
                      << "Warning, creating " << index - _clouds.size()
                      << " empty clouds when inserting to index " << index
                      << ", current size is " << _clouds.size()
                      << "...why not use addCloud?\n";
        _clouds.resize(index + 1);
    }

    _clouds.at(index) = cloud;
} //...CloudManager::setCloud()

DecoratedCloud& CloudManager::getCloud(int index) {
    if (index < _clouds.size())
        return _clouds.at(index);
    else {
        std::cerr << "Cannot return cloud with id " << index
                  << ", only have " << _clouds.size()
                  << " clouds...returning empty cloud\n";
        throw new std::runtime_error("No such cloud");
    }
} //...CloudManager::getCloud()

DecoratedCloud const& CloudManager::getCloud(int index) const {
    return const_cast<DecoratedCloud const&>(
        const_cast<CloudManager*>(this)->getCloud(index)
    );
} //...CloudManager::getCloud() (const)

} //...ns acq