/*
 Copyright 2017 D-Wave Systems Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the Licekonse is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#pragma once
#include "stdheaders_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

extern bool dw_established(void);

extern int dw_init(void);

extern void dw_close(void);

extern void dw_solver(double **val, int maxNodes, int8_t *Q);

#ifdef __cplusplus
}
#endif
