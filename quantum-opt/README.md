## IR Manipulation

The *quantum-opt* build target provides a general IR processing & debugging tool.

The most basic way to process IR is to test whether it can be parsed and re-emitted correctly in a process called "round-tripping".
This will guarantee that the textual format is syntactically correct and can be converted to and from its in-memory representation, as well as that all IR invariants are satisfied by the input.
To do so, run the tool on an `.mlir` input file or stdin via `quantum-opt <input>`.

More importantly, the opt tool can be used to *transform* IR via compiler passes defined in [lib/Transforms](../lib/Transforms/).
In general, passes can be arbitrarily combined to form a pass pipeline via a `quantum-opt -<pass1> -<pass2> ...`, typically with the goal to optimize a program and/or transform it a lower level of abstraction.
Certain passes, however, may only make sense under a certain (partial) ordering.
The following passes are available on quantum programs (in addition to all standard MLIR passes, see `quantum-opt -h` for the full list):

- `-convert-mem-to-val` : Convert quantum operations from memory to value semantics.
- `-quantum-gate-opt` : Run a variety of quantum optimization patterns using the greedy pattern rewrite driver.
- `-circuit-inline` : Inline circuit calls.
- `-count-resources` : Remove all quantum operations from the program & count quantum resources instead.
- `-strip-circ` : Remove unused circuit definitions.
- `-lower-ctrl` : Lower controlled circuit calls by propagating the control modifier into the function body.
