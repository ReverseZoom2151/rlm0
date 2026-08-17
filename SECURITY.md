# Security policy

rlm0 executes model-authored code over user-supplied context. Treat that context
as hostile.

Use `DockerSandbox` or `MicroVMSandbox` for untrusted material. The subprocess
backend is not a security boundary: it runs with your user, filesystem, and
network access. API credentials remain on the host and must never be bound into
the REPL.

To report a vulnerability, open a private GitHub security advisory for
`ReverseZoom2151/rlm0`. Do not include a working exploit in a public issue.
