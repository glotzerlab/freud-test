// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "LinkCell.h"

/*! \file LinkCell.cc
    \brief Build a cell list from a set of points.
*/

namespace freud { namespace locality {

/********************
 * IteratorLinkCell *
 ********************/
void IteratorLinkCell::copy(const IteratorLinkCell& rhs)
{
    m_cell_list = rhs.m_cell_list;
    m_Np = rhs.m_Np;
    m_Nc = rhs.m_Nc;
    m_cur_idx = rhs.m_cur_idx;
    m_cell = rhs.m_cell;
}

bool IteratorLinkCell::atEnd() const
{
    return (m_cur_idx == LINK_CELL_TERMINATOR);
}

unsigned int IteratorLinkCell::next()
{
    m_cur_idx = m_cell_list[m_cur_idx];
    return m_cur_idx;
}

unsigned int IteratorLinkCell::begin()
{
    m_cur_idx = m_Np + m_cell;
    m_cur_idx = m_cell_list[m_cur_idx];
    return m_cur_idx;
}

/*********************
 * IteratorCellShell *
 *********************/

void IteratorCellShell::operator++()
{
    // this bool indicates that we have wrapped over in whichever
    // direction we are looking and should move to the next
    // row/plane
    bool wrapped(false);

    switch (m_stage)
    {
    // +y wedge: iterate over x and (possibly) z
    // zs = list(range(-N + 1, N)) if threeD else [0]
    // for r in itertools.product(range(-N, N), [N], zs):
    //     yield r
    case 0:
        ++m_current_x;
        wrapped = m_current_x >= m_range;
        m_current_x -= 2 * static_cast<int>(wrapped) * m_range;
        if (!m_is2D)
        {
            m_current_z += static_cast<int>(wrapped);
            wrapped = m_current_z >= m_range;
            m_current_z += static_cast<int>(wrapped) * (1 - 2 * m_range);
        }
        if (wrapped)
        {
            ++m_stage;
            m_current_x = m_range;
        }
        break;
        // +x wedge: iterate over y and (possibly) z
        // for r in itertools.product([N], range(N, -N, -1), zs):
        //     yield r
    case 1:
        --m_current_y;
        wrapped = m_current_y <= -m_range;
        m_current_y += 2 * static_cast<int>(wrapped) * m_range;
        if (!m_is2D)
        {
            m_current_z += static_cast<int>(wrapped);
            wrapped = m_current_z >= m_range;
            m_current_z += static_cast<int>(wrapped) * (1 - 2 * m_range);
        }
        if (wrapped)
        {
            ++m_stage;
            m_current_y = -m_range;
        }
        break;
        // -y wedge: iterate over x and (possibly) z
        // for r in itertools.product(range(N, -N, -1), [-N], zs):
        //     yield r
    case 2:
        --m_current_x;
        wrapped = m_current_x <= -m_range;
        m_current_x += 2 * static_cast<int>(wrapped) * m_range;
        if (!m_is2D)
        {
            m_current_z += static_cast<int>(wrapped);
            wrapped = m_current_z >= m_range;
            m_current_z += static_cast<int>(wrapped) * (1 - 2 * m_range);
        }
        if (wrapped)
        {
            ++m_stage;
            m_current_x = -m_range;
        }
        break;
        // -x wedge: iterate over y and (possibly) z
        // for r in itertools.product([-N], range(-N, N), zs):
        //     yield r
    case 3:
        ++m_current_y;
        wrapped = m_current_y >= m_range;
        m_current_y -= 2 * static_cast<int>(wrapped) * m_range;
        if (!m_is2D)
        {
            m_current_z += static_cast<int>(wrapped);
            wrapped = m_current_z >= m_range;
            m_current_z += static_cast<int>(wrapped) * (1 - 2 * m_range);
        }
        if (wrapped)
        {
            if (m_is2D) // we're done for this range
            {
                reset(m_range + 1);
            }
            else
            {
                ++m_stage;
                m_current_x = -m_range;
                m_current_y = -m_range;
                m_current_z = -m_range;
            }
        }
        break;
        // -z face and +z face: iterate over x and y
        // grid = list(range(-N, N + 1))
        // if threeD:
        //     # make front and back in z
        //     for (x, y) in itertools.product(grid, grid):
        //         yield (x, y, N)
        //         if N > 0:
        //             yield (x, y, -N)
        // elif N == 0:
        //     yield (0, 0, 0)
    case 4:
    case 5:
    default:
        ++m_current_x;
        wrapped = m_current_x > m_range;
        m_current_x -= static_cast<int>(wrapped) * (2 * m_range + 1);
        m_current_y += static_cast<int>(wrapped);
        wrapped = m_current_y > m_range;
        m_current_y -= static_cast<int>(wrapped) * (2 * m_range + 1);
        if (wrapped)
        {
            // 2D cases have already moved to the next stage by
            // this point, only deal with 3D
            ++m_stage;
            m_current_z = m_range;

            // if we're done, move on to the next range
            if (m_stage > 5)
            {
                reset(m_range + 1);
            }
        }
        break;
    }
}

void IteratorCellShell::reset(unsigned int range)
{
    // The range is always a positive integer, but since we have to iterate
    // over both positive and negative shells we store m_range as a signed
    // integer.
    m_range = static_cast<int>(range);
    m_stage = 0;
    m_current_x = -m_range;
    m_current_y = m_range;
    if (m_is2D)
    {
        m_current_z = 0;
    }
    else
    {
        m_current_z = -m_range + 1;
    }

    if (range == 0)
    {
        m_current_z = 0;
        // skip to the last stage
        m_stage = 5;
    }
}

/********************
 * LinkCell *
 ********************/

// Default constructor
LinkCell::LinkCell() : NeighborQuery() {}

LinkCell::LinkCell(const box::Box& box, const vec3<float>* points, unsigned int n_points, float cell_width)
    : NeighborQuery(box, points, n_points), m_cell_width(cell_width)
{
    // If no cell width is provided, we calculate the system density and
    // estimate the number of cells that would lead to 10 particles per cell.
    // Want n_points/num_cells = 10
    if (cell_width == 0)
    {
        // This number is arbitrary because there is no way to determine an
        // appropriate cell density for an arbitrary triclinic box.
        const unsigned int num_particle_per_cell = 10;
        const unsigned int desired_num_cells
            = std::max(n_points / num_particle_per_cell, static_cast<unsigned int>(1));
        m_cell_width = std::cbrtf(box.getVolume() / static_cast<float>(desired_num_cells));
    }

    m_celldim = computeDimensions(box, m_cell_width);

    // Check if box is too small!
    vec3<float> nearest_plane_distance = box.getNearestPlaneDistance();
    if ((m_cell_width * 2.0 > nearest_plane_distance.x) || (m_cell_width * 2.0 > nearest_plane_distance.y)
        || (!box.is2D() && m_cell_width * 2.0 > nearest_plane_distance.z))
    {
        throw std::runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
    }
    // Only 1 cell deep in 2D
    if (box.is2D())
    {
        m_celldim.z = 1;
    }

    m_size = m_celldim.x * m_celldim.y * m_celldim.z;
    if (m_size < 1)
    {
        throw std::runtime_error("At least one cell must be present.");
    }

    computeCellList(points, n_points);
}

unsigned int LinkCell::getCellIndex(const vec3<int> cellCoord) const
{
    int w = static_cast<int>(m_celldim.x);
    int h = static_cast<int>(m_celldim.y);
    int d = static_cast<int>(m_celldim.z);

    int x = cellCoord.x % w;
    x += (x < 0 ? w : 0);
    int y = cellCoord.y % h;
    y += (y < 0 ? h : 0);
    int z = cellCoord.z % d;
    z += (z < 0 ? d : 0);

    return coordToIndex(x, y, z);
}

vec3<unsigned int> LinkCell::computeDimensions(const box::Box& box, float cell_width)
{
    vec3<unsigned int> dim;

    vec3<float> L = box.getNearestPlaneDistance();
    dim.x = (unsigned int) ((L.x) / (cell_width));
    dim.y = (unsigned int) ((L.y) / (cell_width));

    if (box.is2D())
    {
        dim.z = 1;
    }
    else
    {
        dim.z = (unsigned int) ((L.z) / (cell_width));
    }

    // In extremely small boxes, the calculated dimensions could go to zero,
    // but need at least one cell in each dimension for particles to be in a
    // cell and to pass the checkCondition tests.
    // Note: freud doesn't actually support these small boxes (as of this
    // writing), but this function will return the correct dimensions
    // required anyways.
    if (dim.x == 0)
    {
        dim.x = 1;
    }
    if (dim.y == 0)
    {
        dim.y = 1;
    }
    if (dim.z == 0)
    {
        dim.z = 1;
    }
    return dim;
}

void LinkCell::computeCellList(const vec3<float>* points, unsigned int n_points)
{
    // determine the number of cells and allocate memory
    unsigned int Nc = getNumCells();
    m_cell_list.prepare(n_points + Nc);
    m_n_points = n_points;

    // initialize memory
    for (unsigned int cell = 0; cell < Nc; cell++)
    {
        m_cell_list[n_points + cell] = LINK_CELL_TERMINATOR;
    }

    // Generate the cell list.
    for (unsigned int i = n_points - 1; i != static_cast<unsigned int>(-1); --i)
    {
        unsigned int cell = getCell(points[i]);
        m_cell_list[i] = m_cell_list[n_points + cell];
        m_cell_list[n_points + cell] = i;
    }
}

vec3<unsigned int> LinkCell::indexToCoord(unsigned int x) const
{
    std::vector<size_t> coord
        = util::ManagedArray<unsigned int>::getMultiIndex({m_celldim.x, m_celldim.y, m_celldim.z}, x);
    // For backwards compatibility with the Index1D layout, the indices and
    // the dimensions are passed in reverse to the indexer. Changing this would
    // also require updating the logic in IteratorCellShell.
    return vec3<unsigned int>(coord[2], coord[1], coord[0]);
}

unsigned int LinkCell::coordToIndex(unsigned int x, unsigned int y, unsigned int z) const
{
    // For backwards compatibility with the Index1D layout, the indices and
    // the dimensions are passed in reverse to the indexer. Changing this would
    // also require updating the logic in IteratorCellShell.
    return util::ManagedArray<unsigned int>::getIndex(
        {m_celldim.z, m_celldim.y, m_celldim.x},
        {static_cast<unsigned int>(z), static_cast<unsigned int>(y), static_cast<unsigned int>(x)});
}

vec3<unsigned int> LinkCell::getCellCoord(const vec3<float>& p) const
{
    vec3<float> alpha = m_box.makeFractional(p);
    vec3<unsigned int> c;
    c.x = (unsigned int) std::floor(alpha.x * float(m_celldim.x));
    c.x %= m_celldim.x;
    c.y = (unsigned int) std::floor(alpha.y * float(m_celldim.y));
    c.y %= m_celldim.y;
    c.z = (unsigned int) std::floor(alpha.z * float(m_celldim.z));
    c.z %= m_celldim.z;
    return c;
}

const std::vector<unsigned int>& LinkCell::getCellNeighbors(unsigned int cell) const
{
    // check if the list of neighbors has been already computed
    // return the list if it has
    // otherwise, compute it and return
    CellNeighbors::const_accessor a;
    if (m_cell_neighbors.find(a, cell))
    {
        return a->second;
    }
    return computeCellNeighbors(cell);
}

const std::vector<unsigned int>& LinkCell::computeCellNeighbors(unsigned int cur_cell) const
{
    std::vector<unsigned int> neighbor_cells;
    vec3<unsigned int> l_idx = indexToCoord(cur_cell);
    const int i = static_cast<int>(l_idx.x);
    const int j = static_cast<int>(l_idx.y);
    const int k = static_cast<int>(l_idx.z);

    // loop over the neighbor cells
    int starti;
    int startj;
    int startk;
    int endi;
    int endj;
    int endk;
    if (m_celldim.x < 3)
    {
        starti = i;
    }
    else
    {
        starti = i - 1;
    }
    if (m_celldim.y < 3)
    {
        startj = j;
    }
    else
    {
        startj = j - 1;
    }
    if (m_celldim.z < 3)
    {
        startk = k;
    }
    else
    {
        startk = k - 1;
    }

    if (m_celldim.x < 2)
    {
        endi = i;
    }
    else
    {
        endi = i + 1;
    }
    if (m_celldim.y < 2)
    {
        endj = j;
    }
    else
    {
        endj = j + 1;
    }
    if (m_celldim.z < 2)
    {
        endk = k;
    }
    else
    {
        endk = k + 1;
    }
    if (m_box.is2D())
    {
        startk = endk = k;
    }

    for (int neighk = startk; neighk <= endk; neighk++)
    {
        for (int neighj = startj; neighj <= endj; neighj++)
        {
            for (int neighi = starti; neighi <= endi; neighi++)
            {
                // wrap back into the box
                unsigned int wrapi = (m_celldim.x + neighi) % m_celldim.x;
                unsigned int wrapj = (m_celldim.y + neighj) % m_celldim.y;
                unsigned int wrapk = (m_celldim.z + neighk) % m_celldim.z;

                unsigned int neigh_cell = coordToIndex(wrapi, wrapj, wrapk);
                // add to the list
                neighbor_cells.push_back(neigh_cell);
            }
        }
    }

    // sort the list
    std::sort(neighbor_cells.begin(), neighbor_cells.end());

    // add the vector of neighbor cells to the hash table
    CellNeighbors::accessor a;
    m_cell_neighbors.insert(a, cur_cell);
    a->second = neighbor_cells;
    return a->second;
}

std::shared_ptr<NeighborQueryPerPointIterator>
LinkCell::querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const
{
    this->validateQueryArgs(args);
    if (args.mode == QueryType::ball)
    {
        return std::make_shared<LinkCellQueryBallIterator>(this, query_point, query_point_idx, args.r_max,
                                                           args.r_min, args.exclude_ii);
    }
    if (args.mode == QueryType::nearest)
    {
        return std::make_shared<LinkCellQueryIterator>(this, query_point, query_point_idx, args.num_neighbors,
                                                       args.r_max, args.r_min, args.exclude_ii);
    }
    throw std::runtime_error("Invalid query mode provided to generic query function.");
}

NeighborBond LinkCellQueryBallIterator::next()
{
    float r_max_sq = m_r_max * m_r_max;
    float r_min_sq = m_r_min * m_r_min;

    vec3<unsigned int> point_cell(m_linkcell->getCellCoord(m_query_point));
    const unsigned int point_cell_index = m_linkcell->getCellIndex(
        vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
    m_searched_cells.insert(point_cell_index);

    // Loop over cell list neighbor shells relative to this point's cell.
    while (true)
    {
        // Iterate over the particles in that cell. Using a local counter
        // variable is safe, because the IteratorLinkCell object is keeping
        // track between calls to next.
        for (unsigned int j = m_cell_iter.next(); !m_cell_iter.atEnd(); j = m_cell_iter.next())
        {
            // Skip ii matches immediately if requested.
            if (m_exclude_ii && m_query_point_idx == j)
            {
                continue;
            }

            const vec3<float> r_ij(m_neighbor_query->getBox().wrap((*m_linkcell)[j] - m_query_point));
            const float r_sq(dot(r_ij, r_ij));

            if (r_sq < r_max_sq && r_sq >= r_min_sq)
            {
                return NeighborBond(m_query_point_idx, j, std::sqrt(r_sq));
            }
        }

        bool out_of_range = false;

        while (true)
        {
            // Determine the next neighbor cell to consider. We're done if we
            // reach a new shell and the closest point of approach to the new
            // shell is greater than our r_max.
            ++m_neigh_cell_iter;

            if (static_cast<float>(m_neigh_cell_iter.getRange() - m_extra_search_width)
                    * m_linkcell->getCellWidth()
                > m_r_max)
            {
                out_of_range = true;
                break;
            }

            const unsigned int neighbor_cell_index = m_linkcell->getCellIndex(
                vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
            // Insertion to an unordered set returns a pair, the second
            // element indicates insertion success or failure (if it
            // already exists)
            if (m_searched_cells.insert(neighbor_cell_index).second)
            {
                // This cell has not been searched yet, so we will iterate
                // over its contents. Otherwise, we loop back, increment
                // the cell shell iterator, and try the next one.
                m_cell_iter = m_linkcell->itercell(neighbor_cell_index);
                break;
            }
        }
        if (out_of_range)
        {
            break;
        }
    }

    m_finished = true;
    return ITERATOR_TERMINATOR;
}

NeighborBond LinkCellQueryIterator::next()
{
    float r_max_sq = m_r_max * m_r_max;
    float r_min_sq = m_r_min * m_r_min;

    vec3<float> plane_distance = m_neighbor_query->getBox().getNearestPlaneDistance();
    float min_plane_distance = std::min(plane_distance.x, plane_distance.y);
    if (!m_neighbor_query->getBox().is2D())
    {
        min_plane_distance = std::min(min_plane_distance, plane_distance.z);
    }
    unsigned int max_range
        = static_cast<unsigned int>(std::ceil(min_plane_distance / (2 * m_linkcell->getCellWidth()))) + 1;

    vec3<unsigned int> point_cell(m_linkcell->getCellCoord(m_query_point));
    const unsigned int point_cell_index = m_linkcell->getCellIndex(
        vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
    m_searched_cells.insert(point_cell_index);

    // Loop over cell list neighbor shells relative to this point's cell.
    if (m_current_neighbors.empty())
    {
        // Expand search cell radius until termination conditions are met.
        while (m_neigh_cell_iter != IteratorCellShell(max_range, m_neighbor_query->getBox().is2D()))
        {
            // Iterate over the particles in that cell. Using a local counter
            // variable is safe, because the IteratorLinkCell object is keeping
            // track between calls to next. However, we have to add an extra
            // check outside to proof ourselves against returning after
            // previous calls to next that have not yet reset the iterator.
            if (!m_cell_iter.atEnd())
            {
                for (unsigned int j = m_cell_iter.next(); !m_cell_iter.atEnd(); j = m_cell_iter.next())
                {
                    // Skip ii matches immediately if requested.
                    if (m_exclude_ii && m_query_point_idx == j)
                    {
                        continue;
                    }
                    const vec3<float> r_ij(m_neighbor_query->getBox().wrap((*m_linkcell)[j] - m_query_point));
                    const float r_sq(dot(r_ij, r_ij));
                    if (r_sq < r_max_sq && r_sq >= r_min_sq)
                    {
                        m_current_neighbors.emplace_back(m_query_point_idx, j, std::sqrt(r_sq));
                    }
                }
            }

            while (true)
            {
                ++m_neigh_cell_iter;

                if (m_neigh_cell_iter == IteratorCellShell(max_range, m_neighbor_query->getBox().is2D()))
                {
                    break;
                }

                const unsigned int neighbor_cell_index = m_linkcell->getCellIndex(
                    vec3<int>(point_cell.x, point_cell.y, point_cell.z) + (*m_neigh_cell_iter));
                // Insertion to an unordered set returns a pair, the second
                // element indicates insertion success or failure (if it
                // already exists)
                if (m_searched_cells.insert(neighbor_cell_index).second)
                {
                    // This cell has not been searched yet, so we will
                    // iterate over its contents. Otherwise, we loop back,
                    // increment the cell shell iterator, and try the next
                    // one.
                    m_cell_iter = m_linkcell->itercell(neighbor_cell_index);
                    break;
                }
            }

            // We can terminate early if we determine when we reach a shell
            // such that we already have k neighbors closer than the
            // closest possible neighbor in the new shell.
            std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
            if ((m_current_neighbors.size() >= m_num_neighbors)
                && (m_current_neighbors[m_num_neighbors - 1].distance
                    < static_cast<float>(m_neigh_cell_iter.getRange() - 1) * m_linkcell->getCellWidth()))
            {
                break;
            }
        }
    }

    while ((m_count < m_num_neighbors) && (m_count < m_current_neighbors.size()))
    {
        m_count++;
        if (m_current_neighbors[m_count - 1].distance > m_r_max)
        {
            m_finished = true;
            return ITERATOR_TERMINATOR;
        }
        return m_current_neighbors[m_count - 1];
    }

    m_finished = true;
    return ITERATOR_TERMINATOR;
}

}; }; // end namespace freud::locality
