# Sandboxing

A sandbox is an isolated place to run code and commands that the model wrote. It has its own disk, no network unless allowed, and hard limits on time, memory and processes. It is disposable: when the run ends, the sandbox is thrown away. Permissions decide whether a command may run at all; the sandbox limits the harm if it does run and turns out to be wrong. This matters because the model can write code nobody has reviewed. A sandbox is not perfect, so secrets must never be placed inside it, and results coming out of it should be treated as untrusted text.
