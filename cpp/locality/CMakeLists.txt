set(VOROPP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/extern/voro++/src")

add_library(
  _locality OBJECT
  AABB.h
  AABBQuery.cc
  AABBQuery.h
  AABBTree.h
  BondHistogramCompute.h
  CMakeLists.txt
  LinkCell.cc
  LinkCell.h
  NeighborBond.h
  NeighborComputeFunctional.cc
  NeighborComputeFunctional.h
  NeighborList.cc
  NeighborList.h
  NeighborPerPointIterator.h
  NeighborQuery.h
  PeriodicBuffer.cc
  PeriodicBuffer.h
  RawPoints.h
  Voronoi.cc
  Voronoi.h
  # For now, compile voro++ object in directly.
  ${VOROPP_SOURCE_DIR}/cell.cc
  ${VOROPP_SOURCE_DIR}/common.cc
  ${VOROPP_SOURCE_DIR}/container.cc
  ${VOROPP_SOURCE_DIR}/unitcell.cc
  ${VOROPP_SOURCE_DIR}/v_compute.cc
  ${VOROPP_SOURCE_DIR}/c_loops.cc
  ${VOROPP_SOURCE_DIR}/v_base.cc
  ${VOROPP_SOURCE_DIR}/wall.cc
  ${VOROPP_SOURCE_DIR}/pre_container.cc
  ${VOROPP_SOURCE_DIR}/container_prd.cc)

# We treat the extern folder as a SYSTEM library to avoid getting any diagnostic
# information from it. In particular, this avoids clang-tidy throwing errors due
# to any issues in external code.
target_include_directories(_locality SYSTEM
                           PUBLIC ${PROJECT_SOURCE_DIR}/extern/)
