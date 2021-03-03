// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "tbb_config.h"

/*! \file tbb_config.cc
    \brief Helper functions to configure tbb
*/

namespace freud { namespace parallel {

tbb::task_scheduler_init* ts = nullptr;

/*! \param N Number of threads to use for TBB computations

    You do not need to call setTBBNumThreads. The default is to use the number of threads in the system. Use
   \a N=0 to set back to the default.

    \note setTBBNumThreads should only be called from the main thread.
*/
void setNumThreads(unsigned int N)
{
    tbb::task_scheduler_init* old_ts(ts);

    if (N == 0)
    {
        N = tbb::task_scheduler_init::automatic;
    }

    delete old_ts;

    // then recreate it
    ts = new tbb::task_scheduler_init(static_cast<int>(N));
}

}; }; // end namespace freud::parallel
