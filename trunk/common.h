/**
 * \file common.h
 * \brief The commonly used header files in project Traffic Classification
 * \author Kefei Lu
 * \sa dataset.h dataset.cpp
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cfloat>

#include <math.h>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <iostream>

#ifdef linux
  #include <stdint.h>
#elif defined(_WIN32)
  // Since MSVC++ doesn't have this standard lib header :)
  #include "stdint.h"
#endif

// Only used in network programming, when processing packet headers.
// #include <netinet/in.h>

#endif
