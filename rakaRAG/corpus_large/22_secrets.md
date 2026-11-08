# Secrets

A secret is a password, key or token. It must never be written into a prompt, a skill or the code. The program loads it from a secret store at run time. Where possible use short-lived credentials issued through a role, so a leaked value stops working soon. A tool receives only the one secret it needs, and the model never sees the value. Logs and traces must redact secrets. A tool server that keeps credentials inside itself, as the Model Context Protocol allows, keeps them away from the agent. Secrets are rotated on a schedule, and immediately after any suspected leak.
