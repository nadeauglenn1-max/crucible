import pytest

from crucible import make, register, registered
from crucible.envs import GuessEnv  # noqa: F401  (import registers "guess"/"sql-task")


def test_known_environments_are_registered():
    assert "guess" in registered()
    assert "sql-task" in registered()


def test_make_reconstructs_from_config():
    original = GuessEnv(low=2, high=9, max_guesses=3)
    rebuilt = make("guess", original.config())
    assert isinstance(rebuilt, GuessEnv)
    assert (rebuilt.low, rebuilt.high, rebuilt.max_guesses) == (2, 9, 3)


def test_register_stamps_env_id():
    assert GuessEnv().name() == "guess"  # the decorator set env_id


def test_make_rejects_unknown_environment():
    with pytest.raises(KeyError):
        make("no-such-env", {})


def test_register_rejects_duplicate_name():
    with pytest.raises(ValueError):

        @register("guess")  # already taken
        class _Dupe:
            pass


def test_make_surfaces_bad_config_as_typeerror():
    with pytest.raises(TypeError):
        make("guess", {"bogus_kwarg": 1})
