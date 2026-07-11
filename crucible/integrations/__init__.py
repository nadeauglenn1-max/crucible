"""Adapters that feed Crucible into specific training stacks.

Each adapter produces the *shape* the target stack expects using only Crucible and
the standard library — so it's usable and testable without installing that stack. The
heavy dependency lives in the user's training environment, not here (the core and
these adapters stay zero-dependency).
"""
