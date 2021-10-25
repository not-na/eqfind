#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  main.py
#  
#  Copyright 2021 notna <notna@apparat.org>
#  
#  This file is part of eqfind.
#
#  eqfind is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  eqfind is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with eqfind.  If not, see <http://www.gnu.org/licenses/>.
#

import eqfind

MAX = 200


def main():
    eqs = eqfind.find_eqs({1, 2, 5, 7}, MAX, p=True)

    for n in range(MAX+1):
        if n not in eqs:
            print(f"{n:>3}: No equations found!")
        else:
            #print(f"{n: 3}: {len(eqs[n])} equations found!")
            print(f"{n:>3}: {len(eqs[n])} equations found! \tSample: {n}={eqs[n].pop()}")
            #for eq in eqs[n]:
            #    print(f"EQ: {n}={eq}")

    print(f"Found equations for {len(eqs.keys())} of {MAX+1} numbers of interest")


if __name__ == "__main__":
    main()
