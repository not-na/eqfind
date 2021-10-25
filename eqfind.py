#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  eqfind.py
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
import itertools
import math
import operator
import os
import time
import multiprocessing.pool
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Union, Callable, Iterator, Optional, Iterable

fact = [math.factorial(i) for i in range(66)]


def factorial(n):
    if n > 64 or n < 0 or not math.isclose(n, int(n)):
        # For speed
        raise OverflowError("Too large, cannot calculate factorial!")
    return fact[int(n)]


OPERATORS_UNARY: dict[str, tuple[Callable, str]] = {
    "": (lambda a: a, "{}"),
#    "sqrt": (math.sqrt, "sqrt({})"),
#    "!": (factorial, "({})!"),
}


def _pow(base, exp):
    if not math.isclose(exp, int(exp)):
        raise ValueError("Non-integer exponents are not supported")
    elif base*exp > 1024:
        raise OverflowError("Too large")
    return base**exp


OPERATORS_BINARY: dict[str, tuple[Callable, str]] = {
    "+": (operator.add, "({}+{})"),
    "-": (operator.sub, "({}-{})"),
    "*": (operator.mul, "({}*{})"),
    "/": (operator.truediv, "({}/{})"),
    "join": (lambda a, b: a*(b//10*10+10)+b, "{}{}"),
#    "pow": (_pow, "({}**{})"),
}

Equation = Union["UnaryEquation", "BinaryEquation", int]

THRESHOLD = 10**(-6)


@dataclass(unsafe_hash=True)
class UnaryEquation:
    operator: str

    a: Equation

    cache: Optional[float] = field(default=None, hash=False)

    @classmethod
    def create(cls, operator: str, a: Equation):
        if operator == "":
            return a
        else:
            return cls(operator=operator, a=a)


@dataclass(unsafe_hash=True)
class BinaryEquation:
    operator: str

    a: Equation

    b: Equation

    cache: Optional[float] = field(default=None, hash=False)

    @classmethod
    def create(cls, operator: str, a: Equation, b: Equation):
        #if operator == "join" and not (isinstance(a, int) and isinstance(b, int)):
        #    raise ValueError("Cannot create join equation from non-integer subequations!")
        #elif operator == "join":
        #    return OPERATORS_BINARY["join"][0](a, b)
        #else:

        return cls(operator=operator, a=a, b=b)


@dataclass
class EqGenPermBase:
    basenums: list[tuple[int, str]]
    perm: tuple[int]

    base_num: int
    base_count: int

    perm_num: int
    perm_count: int

    maxn: int


def run_eq(eq: Equation) -> Optional[float]:
    try:
        return _run_eq(eq)
    except ZeroDivisionError:
        # May happen
        return None
    except ValueError:
        # May happen if factorial gets a non-integer value
        return None
    except OverflowError:
        # Factorials often cause this
        return None


def _run_eq(eq: Equation) -> float:
    # Run equation recursively
    if isinstance(eq, int):
        return eq
    elif isinstance(eq, BinaryEquation):
        if eq.cache is None:
            if eq.operator == "join" and not (isinstance(eq.a, int) and isinstance(eq.b, int)):
                raise ValueError("Cannot create join equation from non-integer subequations!")
            a = run_eq(eq.a)
            b = run_eq(eq.b)
            if a is None or b is None:
                raise ValueError("Cannot calculate with none values!")
            eq.cache = OPERATORS_BINARY[eq.operator][0](a, b)
        return eq.cache
    elif isinstance(eq, UnaryEquation):
        if eq.cache is None:
            a = run_eq(eq.a)
            if a is None:
                raise ValueError("Cannot calculate with none values!")
            eq.cache = OPERATORS_UNARY[eq.operator][0](a)
        return eq.cache


@lru_cache(maxsize=1024)
def format_eq(eq: Equation) -> str:
    # Format equation recursively
    if isinstance(eq, int):
        return str(eq)
    elif isinstance(eq, BinaryEquation):
        return OPERATORS_BINARY[eq.operator][1].format(format_eq(eq.a), format_eq(eq.b))
    elif isinstance(eq, UnaryEquation):
        return OPERATORS_UNARY[eq.operator][1].format(format_eq(eq.a))


def gen_eqs(nums: set[int], p: bool = False) -> Iterator[Equation]:
    # Generates all possible equations from the given set of 4 integers

    # First, generate all permutations (since the equations always use the numbers in order)
    perm: Iterable[int]
    for i, perm in enumerate(itertools.permutations(nums)):
        if p:
            print(f"Permutation: {perm}")

        # Next, generate all base numbers run through all unary operators
        nums_u1 = []
        for n in perm:
            nums_u1.append([])
            for op in OPERATORS_UNARY.keys():
                if op == "":
                    t_eq = n
                else:
                    t_eq = UnaryEquation.create(operator=op, a=n)
                nums_u1[-1].append(t_eq)

        # Use cartesian product to loop through all combinations of base numbers
        basenums: list[Equation]
        for j, basenums in enumerate(itertools.product(*nums_u1)):
            if p:
                print(f"Base: {[format_eq(b) for b in basenums]} (#{j+1} of {len(OPERATORS_UNARY)**len(nums)}) (perm {i+1}/{math.perm(len(nums))})")
            # Next, build up tree-of-trees of equations and return each final one
            for eq in _gen_recursive(basenums):
                yield eq


def gen_eqseeds(nums: set[int], maxn: int = 100) -> Iterator[EqGenPermBase]:
    # Generates all possible equations from the given set of 4 integers

    # First, generate all permutations (since the equations always use the numbers in order)
    perm: Iterable[int]
    for i, perm in enumerate(itertools.permutations(nums)):

        # Next, generate all base numbers run through all unary operators
        nums_u1 = []
        for n in perm:
            nums_u1.append([])
            for op in OPERATORS_UNARY.keys():
                if op == "":
                    t_eq = n
                else:
                    t_eq = UnaryEquation.create(operator=op, a=n)
                nums_u1[-1].append(t_eq)

        # Use cartesian product to loop through all combinations of base numbers
        basenums: list[Equation]
        for j, basenums in enumerate(itertools.product(*nums_u1)):
            b = []
            for e in basenums:
                if isinstance(e, int):
                    b.append((e, ""))
                else:
                    b.append((e.a, e.operator))

            yield EqGenPermBase(
                basenums=b,
                perm=tuple(perm),
                base_num=j+1,
                base_count=len(OPERATORS_UNARY)**len(nums),
                perm_num=i+1,
                perm_count=math.perm(len(nums)),
                maxn=maxn,
            )


def _gen_recursive(eqs: list[Equation]) -> Iterator[Equation]:
    if len(eqs) == 1:
        yield eqs[0]
    elif len(eqs) == 2:
        for op in OPERATORS_BINARY.keys():
            yield BinaryEquation.create(operator=op, a=eqs[0], b=eqs[1])
    else:
        # Two parts: first without ignoring the first term, then with ignoring the first term

        # Part 1: a, b, c, d -> (a, b), (c, d)

        outs = []
        for a, b in zip(eqs[0::2], eqs[1::2]):
            outs.append([])
            for op in OPERATORS_BINARY:
                eq = BinaryEquation.create(operator=op, a=a, b=b)
                for op2 in OPERATORS_UNARY:
                    outs[-1].append(UnaryEquation.create(operator=op2, a=eq))

        # Add the last part if needed
        if len(eqs) % 2 == 1:
            outs.append([eqs[-1]]*(len(OPERATORS_UNARY)*len(OPERATORS_BINARY)))

        for prod in itertools.product(*outs):
            for eq in _gen_recursive(prod):
                yield eq

        # Part 2: a, b, c, d -> a, (b, c), d

        eqs = list(eqs)
        outs = [[eqs.pop(0)]*(len(OPERATORS_UNARY) * len(OPERATORS_BINARY))]
        for a, b in zip(eqs[0::2], eqs[1::2]):
            outs.append([])
            for op in OPERATORS_BINARY:
                eq = BinaryEquation.create(operator=op, a=a, b=b)
                for op2 in OPERATORS_UNARY:
                    outs[-1].append(UnaryEquation.create(operator=op2, a=eq))

        # Add the last part if needed
        if len(eqs) % 2 == 1:
            outs.append([eqs[-1]] * (len(OPERATORS_UNARY) * len(OPERATORS_BINARY)))

        for prod in itertools.product(*outs):
            for eq in _gen_recursive(prod):
                yield eq


def find_eqs(nums: set[int], maxn: int, p: bool = False) -> dict[int: set[str]]:
    out = {}

    n = 0
    t = time.monotonic()
    lt = time.monotonic()
    ln = 0

    for eq in gen_eqs(nums, p=p):
        n += 1
        ct = time.monotonic()
        if p and ct-lt > 1:
            dt = ct-lt
            print(f"Testing equation #{n} ({int((n-ln)/dt)}eqn/s)")
            ln = n
            lt = ct
            #print(f"Testing equation {format_eq(eq)}...")
        num = run_eq(eq)

        if num is not None and 0 <= num <= maxn and math.isclose(num, int(num), rel_tol=THRESHOLD):
            if int(num) not in out:
                out[int(num)] = set()

            feq = format_eq(eq)
            if p and feq not in out[int(num)] and len(out[int(num)]) < 10:
                print(f"Found equation for n={int(num)}: {feq}")
            out[int(num)].add(feq)

    if p:
        print(f"Finished after {time.monotonic()-t}s ({n} equations, {int(n/(time.monotonic()-t))}eqn/s)!")

    return out


def _find_multiproc(seed: EqGenPermBase) -> tuple[dict[int, set[str]], int]:
    basenums = [UnaryEquation.create(e[1], e[0]) for e in seed.basenums]

    print(f"Base: {[format_eq(b) for b in basenums]} (#{seed.base_num} of {seed.base_count}) (perm {seed.perm_num}/{seed.perm_count})")

    out = {}
    n = 0

    for eq in _gen_recursive(basenums):
        n += 1
        num = run_eq(eq)

        if num is not None and 0 <= num <= seed.maxn and math.isclose(num, int(num), rel_tol=THRESHOLD):
            if int(num) not in out:
                out[int(num)] = set()

            feq = format_eq(eq)
            if feq not in out[int(num)] and len(out[int(num)]) < 2:
                print(f"Found equation for n={int(num)}: {feq}")
            out[int(num)].add(feq)

    return out, n


def find_multiproc(nums: set[int], maxn: int, p: bool = False, procs: int = len(os.sched_getaffinity(0))) -> dict[int, set[str]]:
    if p:
        print(f"Starting pool with {procs} processes")

    mapped = []
    with multiprocessing.Pool(procs) as pool:
        if p:
            print("Pool started. Starting work.")

        t = time.monotonic()
        mapped = pool.map(_find_multiproc, list(gen_eqseeds(nums, maxn)))

    if p:
        print("Waiting for join...")
    pool.close()
    pool.join()

    et = time.monotonic()
    if p:
        print(f"Finished in {et-t:.02}s! Now combining results")

    out = {}

    on = 0
    for output, n in mapped:
        on += n
        for k, v in output.items():
            if k not in out:
                out[k] = set()

            out[k].update(v)

    if p:
        print(f"Evaluated {on} equations ({int(on/(et-t))}eqn/s)")

    return out
