/* camellia.h	ver 1.2.0
 *
 * Copyright (C) 2006,2007
 * NTT (Nippon Telegraph and Telephone Corporation).
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifndef HEADER_CAMELLIA_H
#define HEADER_CAMELLIA_H

#include "cu_ocb/constants.h"

namespace cu_ocb {

__global__ void camellia_setup128(const unsigned char *key, u32 *_subkey);

__global__ void camellia_encrypt128(const u32 *_subkey, const u32 *offsets, const u32 *in_buf, u32 *out_buf);

__global__ void camellia_decrypt128(const u32 *_subkey, const u32 *offsets, const u32 *in_buf, u32* out_buf);

}

#endif /* HEADER_CAMELLIA_H */
