# eqfind

This is a little project designed to find equations that can be evaluated to a specific
number. It can utilize all cores via the multiprocessing module.

While some effort was put into optimization (mostly caching and cutoffs for large numbers),
many things remain largely unoptimized. In particular, identical equations could be pre-filtered
to avoid wasting time on them. This would require major changes in the equation generation
code and was thus not done.

Equations are generated in a fairly memory efficient manner using generators.
To speed up evaluation, the equations are generated in such a way as to allow
reuse of parts of the equation without needing to re-evaluate it.

Improvements and pull requests are welcome.

Inspired by [this reddit post](https://www.reddit.com/r/mildlyinteresting/comments/q66jb5/i_created_this_puzzle_where_using_only_4_digits/).
