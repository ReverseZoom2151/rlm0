# Security policy

rlm0 executes model-authored code over user-supplied context. Treat that context
as hostile.

Use `DockerSandbox` or `MicroVMSandbox` for untrusted material. The subprocess
backend is not a security boundary: it runs with your user, filesystem, and
network access. API credentials remain on the host and must never be bound into
the REPL.

`MicroVMSandbox` is experimental. It currently requires Docker plus a registered
OCI microVM runtime and checks that the guest reports a kernel distinct from the
host. Do not assume Podman or libkrun support from this project. Docker is a
useful containment layer but shares the host kernel; use the microVM backend on
a host you control when the context is genuinely hostile.

The default Docker image is pinned by manifest digest. Review and update that
digest as an explicit dependency change, then exercise the sandbox tests against
the replacement before using it for untrusted context.

If a sandbox reports malformed protocol data, loses its process, or cannot
prove a requested isolation property, treat the run as failed. It must never
fall back to subprocess execution.

To report a vulnerability, open a private GitHub security advisory for
`ReverseZoom2151/rlm0`. Do not include a working exploit in a public issue.
