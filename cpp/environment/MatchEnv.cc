// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <sstream>
#include <stdexcept>

#include "MatchEnv.h"

#include "NeighborBond.h"
#include "NeighborComputeFunctional.h"

namespace freud { namespace environment {

/*****************
 * EnvDisjoinSet *
 *****************/
EnvDisjointSet::EnvDisjointSet(unsigned int Np) : rank(std::vector<unsigned int>(Np, 0)), m_max_num_neigh(0)
{}

void EnvDisjointSet::merge(const unsigned int a, const unsigned int b,
                           BiMap<unsigned int, unsigned int> vec_map, rotmat3<float>& rotation)
{
    // if tree heights are equal, merge b to a
    if (rank[s[a].env_ind] == rank[s[b].env_ind])
    {
        // Get the ENTIRE set that corresponds to head_b.
        unsigned int head_b = find(b);
        std::vector<unsigned int> m_set = findSet(head_b);
        for (unsigned int node : m_set)
        {
            // Go through the entire tree/set.
            // Make a copy of the old set of vector indices for this
            // particular node.
            std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

            // Set the vector indices properly.
            // Take the LEFT MAP view of the proper_a<->proper_b bimap.
            // Iterate over the values of proper_a_ind IN ORDER, find the
            // value of proper_b_ind that corresponds to each proper_a_ind,
            // and set it properly.
            for (unsigned int proper_a_ind = 0; proper_a_ind < vec_map.size(); proper_a_ind++)
            {
                unsigned int proper_b_ind = vec_map.left[proper_a_ind];

                // old_node_vec_ind[proper_b_ind] is "relative_b_ind"
                s[node].vec_ind[proper_a_ind] = old_node_vec_ind[proper_b_ind];
            }

            // set the environment index properly
            s[node].env_ind = s[a].env_ind;

            // set the proper orientation. ORDER MATTERS since rotations
            // don't commute in 3D.
            s[node].proper_rot = rotation * s[node].proper_rot;

            // we've added another leaf to the tree or whatever the lingo is.
            rank[s[a].env_ind]++;
        }
    }
    else
    {
        // merge the shorter tree to the taller one
        if (rank[s[a].env_ind] > rank[s[b].env_ind])
        {
            // Get the ENTIRE set that corresponds to head_b.
            unsigned int head_b = find(b);
            std::vector<unsigned int> m_set = findSet(head_b);
            for (unsigned int node : m_set)
            {
                // Go through the entire tree/set.
                // Make a copy of the old set of vector indices for this
                // particular node. This is complicated and weird.
                std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

                // Set the vector indices properly.
                // Take the LEFT MAP view of the proper_a<->proper_b bimap.
                // Iterate over the values of proper_a_ind IN ORDER, find the
                // value of proper_b_ind that corresponds to each proper_a_ind,
                // and set it properly.
                for (unsigned int proper_a_ind = 0; proper_a_ind < vec_map.size(); proper_a_ind++)
                {
                    unsigned int proper_b_ind = vec_map.left[proper_a_ind];

                    // old_node_vec_ind[proper_b_ind] is "relative_b_ind"
                    s[node].vec_ind[proper_a_ind] = old_node_vec_ind[proper_b_ind];
                }
                // set the environment index properly
                s[node].env_ind = s[a].env_ind;

                // set the proper orientation. ORDER MATTERS since rotations
                // don't commute in 3D.
                s[node].proper_rot = rotation * s[node].proper_rot;

                // we've added another leaf to the tree or whatever the lingo is.
                rank[s[a].env_ind]++;
            }
        }
        else
        {
            rotmat3<float> rotationT = transpose(rotation);
            // Get the ENTIRE set that corresponds to head_a.
            unsigned int head_a = find(a);
            std::vector<unsigned int> m_set = findSet(head_a);
            for (unsigned int node : m_set)
            {
                // Go through the entire tree/set.
                // Make a copy of the old set of vector indices for this
                // particular node. This is complicated and weird.
                std::vector<unsigned int> old_node_vec_ind = s[node].vec_ind;

                // Set the vector indices properly.
                // Take the RIGHT MAP view of the proper_a<->proper_b bimap.
                // Iterate over the values of proper_b_ind IN ORDER, find the
                // value of proper_a_ind that corresponds to each proper_b_ind,
                // and set it properly.
                for (unsigned int proper_b_ind = 0; proper_b_ind < vec_map.size(); proper_b_ind++)
                {
                    unsigned int proper_a_ind = vec_map.right[proper_b_ind];

                    // old_node_vec_ind[proper_a_ind] is "relative_a_ind"
                    s[node].vec_ind[proper_b_ind] = old_node_vec_ind[proper_a_ind];
                }

                // set the environment index properly
                s[node].env_ind = s[b].env_ind;

                // set the proper orientation. ORDER MATTERS since rotations
                // don't commute in 3D.
                // note that here we are rotating vector set proper_a such
                // that it matches vector set proper_b, so we need to multiply
                // by the INVERSE (transpose) of the matrix rotation.
                s[node].proper_rot = rotationT * s[node].proper_rot;

                // we've added another leaf to the tree or whatever the lingo is.
                rank[s[b].env_ind]++;
            }
        }
    }
}

unsigned int EnvDisjointSet::find(const unsigned int c)
{
    unsigned int r = c;

    // follow up to the head of the tree
    while (s[r].env_ind != r)
    {
        r = s[r].env_ind;
    }

    // path compression
    unsigned int i = c;
    while (i != r)
    {
        unsigned int j = s[i].env_ind;
        s[i].env_ind = r;
        i = j;
    }
    return r;
}

std::vector<unsigned int> EnvDisjointSet::findSet(const unsigned int m)
{
    bool invalid_ind = true;

    std::vector<unsigned int> m_set;

    // this is wildly inefficient
    for (unsigned int i = 0; i < s.size(); i++)
    {
        // get the head environment index
        unsigned int head_env = find(s[i].env_ind);
        // if we are part of the environment m, add the vectors to m_set
        if (head_env == m)
        {
            m_set.push_back(i);
            invalid_ind = false;
        }
    }

    if (invalid_ind)
    {
        std::ostringstream msg;
        msg << "Index " << m << " must be a head index in the environment set!" << std::endl;
        throw std::invalid_argument(msg.str());
    }

    return m_set;
}

std::vector<vec3<float>> EnvDisjointSet::getAvgEnv(const unsigned int m)
{
    bool invalid_ind = true;

    std::vector<vec3<float>> env(m_max_num_neigh, vec3<float>(0.0, 0.0, 0.0));
    unsigned int N = 0;

    // loop over all the environments in the set
    for (auto& i : s)
    {
        // if this environment is NOT a ghost (i.e. non-physical):
        if (!i.ghost)
        {
            // get the head environment index
            unsigned int head_env = find(i.env_ind);
            // if we are part of the environment m, add the vectors to env
            if (head_env == m)
            {
                // loop through the vectors, getting them properly indexed
                // add them to env
                for (unsigned int proper_ind = 0; proper_ind < i.vecs.size(); proper_ind++)
                {
                    unsigned int relative_ind = i.vec_ind[proper_ind];
                    env[proper_ind] += i.proper_rot * i.vecs[relative_ind];
                }
                ++N;
                invalid_ind = false;
            }
        }
    }

    if (invalid_ind)
    {
        std::ostringstream msg;
        msg << "Index " << m << " must be a head index in the environment set!" << std::endl;
        throw std::invalid_argument(msg.str());
    }

    // loop through the vectors in env now, dividing by the total number
    // of contributing particle environments to make an average
    for (unsigned int n = 0; n < m_max_num_neigh; n++)
    {
        vec3<float> normed = env[n] / static_cast<float>(N);
        env[n] = normed;
    }
    return env;
}

std::vector<vec3<float>> EnvDisjointSet::getIndividualEnv(const unsigned int m)
{
    if (m >= s.size())
    {
        std::ostringstream msg;
        msg << "Index " << m << " must be less than the size of the environment set!" << std::endl;
        throw std::invalid_argument(msg.str());
    }

    std::vector<vec3<float>> env;
    for (unsigned int n = 0; n < m_max_num_neigh; n++)
    {
        env.emplace_back(0.0, 0.0, 0.0);
    }

    // loop through the vectors, getting them properly indexed
    // add them to env
    for (unsigned int proper_ind = 0; proper_ind < s[m].vecs.size(); proper_ind++)
    {
        unsigned int relative_ind = s[m].vec_ind[proper_ind];
        env[proper_ind] += s[m].proper_rot * s[m].vecs[relative_ind];
    }

    return env;
}

/*************************
 * Convenience functions *
 *************************/
std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> isSimilar(Environment& e1, Environment& e2,
                                                                       float threshold_sq, bool registration)
{
    BiMap<unsigned int, unsigned int> vec_map;
    rotmat3<float> rotation = rotmat3<float>(); // this initializes to the identity matrix

    // If the vector sets do not have equal numbers of vectors, just return
    // an empty map since the 1-1 bimapping will be too weird in this case.
    if (e1.vecs.size() != e2.vecs.size())
    {
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>(rotation, vec_map);
    }

    std::vector<vec3<float>> v1(e1.vecs.size());
    std::vector<vec3<float>> v2(e2.vecs.size());

    // get the vectors into the proper orientation and order with respect to
    // their parent environment
    for (unsigned int m = 0; m < e1.vecs.size(); m++)
    {
        v1[m] = e1.proper_rot * e1.vecs[e1.vec_ind[m]];
        v2[m] = e2.proper_rot * e2.vecs[e2.vec_ind[m]];
    }

    // If we have to register, first find the rotated set of v2 that best maps
    // to v1. The Fit operation CHANGES v2.
    if (registration)
    {
        RegisterBruteForce r = RegisterBruteForce(v1);
        r.Fit(v2);
        // get the optimal rotation to take v2 to v1
        std::vector<vec3<float>> rot = r.getRotation();
        // rot must be a 3x3 matrix. if it isn't, something has gone wrong.
        rotation = rotmat3<float>(rot[0], rot[1], rot[2]);
        BiMap<unsigned int, unsigned int> tmp_vec_map = r.getVecMap();

        for (const auto* registered_pair : tmp_vec_map)
        {
            // RegisterBruteForce has found the vector mapping that results in
            // minimal RMSD, as best as it can figure out.
            // Does this vector mapping pass the more stringent criterion
            // imposed by the threshold?
            vec3<float> delta = v1[registered_pair->first] - v2[registered_pair->second];
            float r_sq = dot(delta, delta);
            if (r_sq < threshold_sq)
            {
                vec_map.emplace(registered_pair->first, registered_pair->second);
            }
        }
    }

    // if we didn't have to register, compare all combinations of vectors
    else
    {
        for (unsigned int i = 0; i < e1.vecs.size(); i++)
        {
            for (unsigned int j = 0; j < e2.vecs.size(); j++)
            {
                vec3<float> delta = v1[i] - v2[j];
                float r_sq = dot(delta, delta);
                if (r_sq < threshold_sq)
                {
                    // these vectors are deemed "matching"
                    // since this is a bimap, this (i,j) pair is only inserted
                    // if j has not already been assigned an i pairing.
                    // (ditto with i not being assigned a j pairing)
                    vec_map.emplace(i, j);
                }
            }
        }
    }

    // if every vector has been paired with every other vector, return this bimap
    if (vec_map.size() == e1.vecs.size())
    {
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>(rotation, vec_map);
    }
    // otherwise, return an empty bimap
    BiMap<unsigned int, unsigned int> empty_map;
    return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>(rotation, empty_map);
}

std::map<unsigned int, unsigned int> isSimilar(const box::Box& box, const vec3<float>* refPoints1,
                                               vec3<float>* refPoints2, unsigned int numRef,
                                               float threshold_sq, bool registration)
{
    Environment e0;
    Environment e1;
    std::tie(e0, e1) = makeEnvironments(box, refPoints1, refPoints2, numRef);

    // call isSimilar for e0 and e1
    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> mapping
        = isSimilar(e0, e1, threshold_sq, registration);
    rotmat3<float> rotation = mapping.first;
    BiMap<unsigned int, unsigned int> vec_map = mapping.second;

    // update refPoints2 in case registration has taken place
    for (unsigned int i = 0; i < numRef; i++)
    {
        refPoints2[i] = rotation * e1.vecs[i];
    }

    // Convert BiMap to a std::map and return
    return vec_map.asMap();
}

std::pair<Environment, Environment> makeEnvironments(const box::Box& box, const vec3<float>* refPoints1,
                                                     vec3<float>* refPoints2, unsigned int numRef)
{
    // create the environment characterized by refPoints1. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have
    // actually encountered in the simulation.
    Environment e0 = Environment(true);

    // create the environment characterized by refPoints2. Index it as 1.
    // set the IGNORE flag to true again.
    Environment e1 = Environment(true);
    e1.env_ind = 1;

    // loop through all the vectors in refPoints1 and refPoints2 and add them
    // to the environments.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < numRef; i++)
    {
        vec3<float> p0 = box.wrap(refPoints1[i]);
        vec3<float> p1 = box.wrap(refPoints2[i]);
        e0.addVec(p0);
        e1.addVec(p1);
    }
    return std::pair<Environment, Environment>(e0, e1);
}

std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> minimizeRMSD(Environment& e1, Environment& e2,
                                                                          float& min_rmsd, bool registration)
{
    BiMap<unsigned int, unsigned int> vec_map;
    rotmat3<float> rotation = rotmat3<float>(); // this initializes to the identity matrix

    // If the vector sets do not have equal numbers of vectors, force the map
    // to be empty since it can never be 1-1.
    // Return the empty vec_map and the identity matrix, and minRMSD = -1.
    if (e1.vecs.size() != e2.vecs.size())
    {
        min_rmsd = -1.0;
        return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>(rotation, vec_map);
    }

    std::vector<vec3<float>> v1(e1.vecs.size());
    std::vector<vec3<float>> v2(e2.vecs.size());

    // Get the vectors into the proper orientation and order with respect
    // to their parent environment
    for (unsigned int m = 0; m < e1.vecs.size(); m++)
    {
        v1[m] = e1.proper_rot * e1.vecs[e1.vec_ind[m]];
        v2[m] = e2.proper_rot * e2.vecs[e2.vec_ind[m]];
    }

    // call RegisterBruteForce::Fit and update min_rmsd accordingly
    RegisterBruteForce r = RegisterBruteForce(v1);
    // If we have to register, first find the rotated set of v2 that best
    // maps to v1. The Fit operation CHANGES v2.
    if (registration)
    {
        r.Fit(v2);
        // get the optimal rotation to take v2 to v1
        std::vector<vec3<float>> rot = r.getRotation();
        // rot must be a 3x3 matrix. if it isn't, something has gone wrong.
        rotation = rotmat3<float>(rot[0], rot[1], rot[2]);
        min_rmsd = r.getRMSD();
        vec_map = r.getVecMap();
    }
    else
    {
        // this will populate vec_map with the correct mapping
        min_rmsd = r.AlignedRMSDTree(makeEigenMatrix(v2), vec_map);
    }

    // return the rotation matrix and bimap
    return std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>>(rotation, vec_map);
}

std::map<unsigned int, unsigned int> minimizeRMSD(const box::Box& box, const vec3<float>* refPoints1,
                                                  vec3<float>* refPoints2, unsigned int numRef,
                                                  float& min_rmsd, bool registration)
{
    Environment e0;
    Environment e1;
    std::tie(e0, e1) = makeEnvironments(box, refPoints1, refPoints2, numRef);

    float tmp_min_rmsd = -1.0;
    std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> mapping
        = minimizeRMSD(e0, e1, tmp_min_rmsd, registration);
    rotmat3<float> rotation = mapping.first;
    BiMap<unsigned int, unsigned int> vec_map = mapping.second;
    min_rmsd = tmp_min_rmsd;

    // update refPoints2 in case registration has taken place
    for (unsigned int i = 0; i < numRef; i++)
    {
        refPoints2[i] = rotation * e1.vecs[i];
    }

    // Convert BiMap to a std::map and return
    return vec_map.asMap();
}

/************
 * MatchEnv *
 ************/
MatchEnv::MatchEnv() = default;

MatchEnv::~MatchEnv() = default;

/**********************
 * EnvironmentCluster *
 **********************/

EnvironmentCluster::~EnvironmentCluster() = default;

Environment MatchEnv::buildEnv(const freud::locality::NeighborQuery* nq,
                               const freud::locality::NeighborList* nlist, size_t num_bonds, size_t& bond,
                               unsigned int i, unsigned int env_ind)
{
    Environment ei = Environment();
    // set the environment index equal to the particle index
    ei.env_ind = env_ind;

    for (; bond < num_bonds && nlist->getNeighbors()(bond, 0) == i; ++bond)
    {
        // compute vec{r} between the two particles
        const size_t j(nlist->getNeighbors()(bond, 1));
        if (i != j)
        {
            vec3<float> delta(bondVector(locality::NeighborBond(i, j), nq, nq->getPoints()));
            ei.addVec(delta);
        }
    }

    return ei;
}

void EnvironmentCluster::compute(const freud::locality::NeighborQuery* nq,
                                 const freud::locality::NeighborList* nlist_arg, locality::QueryArgs qargs,
                                 const freud::locality::NeighborList* env_nlist_arg,
                                 locality::QueryArgs env_qargs, float threshold, bool registration,
                                 bool global)
{
    const locality::NeighborList nlist
        = locality::makeDefaultNlist(nq, nlist_arg, nq->getPoints(), nq->getNPoints(), qargs);
    const locality::NeighborList env_nlist
        = locality::makeDefaultNlist(nq, env_nlist_arg, nq->getPoints(), nq->getNPoints(), env_qargs);

    unsigned int Np = nq->getNPoints();
    m_env_index.prepare(Np);

    float m_threshold_sq = threshold * threshold;

    nlist.validate(Np, Np);
    env_nlist.validate(Np, Np);
    size_t env_bond(0);
    const size_t env_num_bonds(env_nlist.getNumBonds());

    // create a disjoint set where all particles belong in their own cluster
    EnvDisjointSet dj(Np);

    // add all the environments to the set
    // take care, here: set things up s.t. the env_ind of every environment
    // matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < Np; i++)
    {
        Environment ei = buildEnv(nq, &env_nlist, env_num_bonds, env_bond, i, i);
        dj.s.push_back(ei);
        dj.m_max_num_neigh = std::max(dj.m_max_num_neigh, ei.num_vecs);
        ;
    }

    // reallocate the m_point_environments array
    m_point_environments.prepare({Np, dj.m_max_num_neigh});

    size_t bond(0);
    // loop through points
    for (unsigned int i = 0; i < Np; i++)
    {
        if (!global)
        {
            // loop over the neighbors
            for (; bond < nlist.getNumBonds() && nlist.getNeighbors()(bond, 0) == i; ++bond)
            {
                const size_t j(nlist.getNeighbors()(bond, 1));
                std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> mapping
                    = isSimilar(dj.s[i], dj.s[j], m_threshold_sq, registration);
                rotmat3<float> rotation = mapping.first;
                BiMap<unsigned int, unsigned int> vec_map = mapping.second;
                // if the mapping between the vectors of the environments
                // is NOT empty, then the environments are similar, so
                // merge them.
                if (!vec_map.empty())
                {
                    // merge the two sets using the disjoint set
                    unsigned int a = dj.find(i);
                    unsigned int b = dj.find(j);
                    if (a != b)
                    {
                        dj.merge(i, j, vec_map, rotation);
                    }
                }
            }
        }
        else
        {
            // loop over all other particles
            for (unsigned int j = i + 1; j < Np; j++)
            {
                std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> mapping
                    = isSimilar(dj.s[i], dj.s[j], m_threshold_sq, registration);
                rotmat3<float> rotation = mapping.first;
                BiMap<unsigned int, unsigned int> vec_map = mapping.second;
                // if the mapping between the vectors of the environments
                // is NOT empty, then the environments are similar, so
                // merge them.
                if (!vec_map.empty())
                {
                    // merge the two sets using the disjoint set
                    unsigned int a = dj.find(i);
                    unsigned int b = dj.find(j);
                    if (a != b)
                    {
                        dj.merge(i, j, vec_map, rotation);
                    }
                }
            }
        }
    }

    // done looping over points. All clusters are now determined. Renumber
    // them from zero to num_clusters-1.
    m_num_clusters = populateEnv(dj);
}

unsigned int EnvironmentCluster::populateEnv(EnvDisjointSet dj)
{
    std::map<unsigned int, unsigned int> label_map;
    std::map<unsigned int, std::vector<vec3<float>>> cluster_env;

    // loop over all environments
    unsigned int label_ind;
    unsigned int cur_set = 0;
    unsigned int particle_ind = 0;
    for (unsigned int i = 0; i < dj.s.size(); i++)
    {
        // only count this if the environment is physical
        if (!dj.s[i].ghost)
        {
            // grab the set of vectors that define this individual environment
            std::vector<vec3<float>> part_vecs = dj.getIndividualEnv(i);

            unsigned int c = dj.find(i);
            // insert the set into the mapping if we haven't seen it before.
            // also grab the vectors that define the set and insert them into cluster_env
            if (label_map.count(c) == 0)
            {
                label_map[c] = cur_set;
                std::vector<vec3<float>> vecs = dj.getAvgEnv(c);
                label_ind = label_map[c];
                cluster_env[label_ind] = vecs;
                cur_set++;
            }
            else
            {
                label_ind = label_map[c];
            }

            // label this particle in m_env_index
            m_env_index[particle_ind] = label_ind;
            for (unsigned int m = 0; m < part_vecs.size(); m++)
            {
                m_point_environments(particle_ind, m) = part_vecs[m];
            }
            particle_ind++;
        }
    }

    // Now update the vector of environments from the map.
    m_cluster_environments.resize(cluster_env.size());
    for (const auto& it : cluster_env)
    {
        m_cluster_environments[it.first] = it.second;
    }

    // specify the number of cluster environments
    return cur_set;
}

/*************************
 * EnvironmentMotifMatch *
 *************************/
void EnvironmentMotifMatch::compute(const freud::locality::NeighborQuery* nq,
                                    const freud::locality::NeighborList* nlist_arg, locality::QueryArgs qargs,
                                    const vec3<float>* motif, unsigned int motif_size, float threshold,
                                    bool registration)
{
    const locality::NeighborList nlist
        = locality::makeDefaultNlist(nq, nlist_arg, nq->getPoints(), nq->getNPoints(), qargs);

    unsigned int Np = nq->getNPoints();
    float m_threshold_sq = threshold * threshold;

    nlist.validate(Np, Np);

    // create a disjoint set where all particles belong in their own cluster.
    // this has to have ONE MORE environment than there are actual particles,
    // because we're inserting the motif into it.
    EnvDisjointSet dj(Np + 1);
    dj.m_max_num_neigh = motif_size;

    // reallocate the m_point_environments array
    m_point_environments.prepare({Np, motif_size});

    // create the environment characterized by motif. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we have
    // actually encountered in the simulation.
    Environment e0 = Environment(true);

    // loop through all the vectors in motif and add them to the environment.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < motif_size; i++)
    {
        vec3<float> p = nq->getBox().wrap(motif[i]);
        e0.addVec(p);
    }

    // add this environment to the set
    dj.s.push_back(e0);

    size_t bond(0);
    const size_t num_bonds(nlist.getNumBonds());

    m_matches.prepare(Np);

    // loop through the particles and add their environments to the set
    // take care, here: set things up s.t. the env_ind of every environment
    // matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < Np; i++)
    {
        unsigned int dummy = i + 1;
        Environment ei = buildEnv(nq, &nlist, num_bonds, bond, i, dummy);
        dj.s.push_back(ei);

        // if the environment matches e0, merge it into the e0 environment set
        std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> mapping
            = isSimilar(dj.s[0], dj.s[dummy], m_threshold_sq, registration);
        rotmat3<float> rotation = mapping.first;
        BiMap<unsigned int, unsigned int> vec_map = mapping.second;
        // if the mapping between the vectors of the environments is NOT empty,
        // then the environments are similar.
        if (!vec_map.empty())
        {
            dj.merge(0, dummy, vec_map, rotation);
            m_matches[i] = true;
        }
        // grab the set of vectors that define this individual environment
        std::vector<vec3<float>> part_vecs = dj.getIndividualEnv(dummy);

        for (unsigned int m = 0; m < part_vecs.size(); m++)
        {
            m_point_environments(i, m) = part_vecs[m];
        }
    }
}

/****************************
 * EnvironmentRMSDMinimizer *
 ****************************/
void EnvironmentRMSDMinimizer::compute(const freud::locality::NeighborQuery* nq,
                                       const freud::locality::NeighborList* nlist_arg,
                                       locality::QueryArgs qargs, const vec3<float>* motif,
                                       unsigned int motif_size, bool registration)
{
    const locality::NeighborList nlist
        = locality::makeDefaultNlist(nq, nlist_arg, nq->getPoints(), nq->getNPoints(), qargs);

    unsigned int Np = nq->getNPoints();

    // create a disjoint set where all particles belong in their own cluster.
    // this has to have ONE MORE environment than there are actual particles,
    // because we're inserting the motif into it.
    EnvDisjointSet dj(Np + 1);
    dj.m_max_num_neigh = motif_size;

    // reallocate the m_point_environments array
    m_point_environments.prepare({Np, motif_size});

    // create the environment characterized by motif. Index it as 0.
    // set the IGNORE flag to true, since this is not an environment we
    // have actually encountered in the simulation.
    Environment e0 = Environment(true);

    // loop through all the vectors in motif and add them to the environment.
    // wrap all the vectors back into the box. I think this is necessary since
    // all the vectors that will be added to actual particle environments will
    // be wrapped into the box as well.
    for (unsigned int i = 0; i < motif_size; i++)
    {
        vec3<float> p = nq->getBox().wrap(motif[i]);
        e0.addVec(p);
    }

    // add this environment to the set
    dj.s.push_back(e0);

    size_t bond(0);
    const size_t num_bonds(nlist.getNumBonds());

    m_rmsds.prepare(Np);

    // loop through the particles and add their environments to the set
    // take care, here: set things up s.t. the env_ind of every environment
    // matches its location in the disjoint set.
    // if you don't do this, things will get screwy.
    for (unsigned int i = 0; i < Np; i++)
    {
        unsigned int dummy = i + 1;
        Environment ei = buildEnv(nq, &nlist, num_bonds, bond, i, dummy);
        dj.s.push_back(ei);

        // if the environment matches e0, merge it into the e0 environment set
        float min_rmsd = -1.0;
        std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> mapping
            = minimizeRMSD(dj.s[0], dj.s[dummy], min_rmsd, registration);
        rotmat3<float> rotation = mapping.first;
        BiMap<unsigned int, unsigned int> vec_map = mapping.second;
        // populate the min_rmsd vector
        m_rmsds[i] = min_rmsd;

        // if the mapping between the vectors of the environments is NOT
        // empty, then the environments are similar.
        // minimizeRMSD should always return a non-empty vec_map, except if
        // e0 and e1 have different numbers of vectors.
        if (!vec_map.empty())
        {
            dj.merge(0, dummy, vec_map, rotation);
        }

        // grab the set of vectors that define this individual environment
        std::vector<vec3<float>> part_vecs = dj.getIndividualEnv(dummy);

        for (unsigned int m = 0; m < part_vecs.size(); m++)
        {
            m_point_environments(i, m) = part_vecs[m];
        }
    }
}

}; }; // end namespace freud::environment
