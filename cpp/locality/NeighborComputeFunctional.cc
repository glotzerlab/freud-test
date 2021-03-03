#include "NeighborComputeFunctional.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for generic looping over neighbors and applying a compute function.
*/

namespace freud { namespace locality {

NeighborList makeDefaultNlist(const NeighborQuery* nq, const NeighborList* nlist,
                              const vec3<float>* query_points, unsigned int num_query_points,
                              locality::QueryArgs qargs)
{
    bool requires_delete(false);
    if (nlist == nullptr)
    {
        auto nqiter(nq->query(query_points, num_query_points, qargs));
        nlist = nqiter->toNeighborList();
        requires_delete = true;
    }
    // Ideally we wouldn't allocate a new NeighborList at all, but we don't
    // want to force calling code to manage the memory of a raw pointer, so we
    // prefer to return by value here. If we returned the nlist directly, the
    // calling code would have to know whether or not the nlist required
    // deleting (i.e. by checking for a null nlist internally). Since the
    // pointer being passed from Cython (if not NULL) will always be a raw
    // pointer, and toNeighborList must also always return a raw pointer (to
    // work with the Cython API of freud) we can't wrap the returned nlist in a
    // smart pointer here (otherwise it will get double freed if it is a
    // user-provided nlist), so using smart pointers here is also not an
    // option. Therefore, this solution of making a copy on the stack is the
    // best option for now.
    locality::NeighborList new_nlist = NeighborList(*nlist);
    new_nlist.validate(num_query_points, nq->getNPoints());
    if (requires_delete)
    {
        delete nlist;
    }
    return new_nlist;
}

}; }; // end namespace freud::locality
