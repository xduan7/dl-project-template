from typing import List
from difflib import get_close_matches


def get_closest_match(
        word: str,
        possibilities: List[str],
        cutoff: float = 0.5,
) -> str:
    """Get the closest match to a word from possibilities.

    Args:
        word: A string as input for matching against all possibilities.
        possibilities: A list of possible strings to match against the
            input string, in which the closest match will be returned.
        cutoff: The similarity cutoff in the range of [0, 1].
            Possibilities that don’t score at least that similar to
            word are ignored.

    Returns:
        The closest match to the `input` word in `possibilities` with
        similarity of at least `cutoff`.

    Raises:
        ValueError: No match from `possibilities` with similarity of
            at least `cutoff` to `word`.

    """
    _lower_case_to_original_possibility_dict = \
        {__p.lower(): __p for __p in possibilities}
    _lower_case_possibilities = \
        _lower_case_to_original_possibility_dict.keys()
    try:
        _lower_case_closest_match = get_close_matches(
            word=word.lower(),
            possibilities=_lower_case_possibilities,
            n=1,
            cutoff=cutoff,
        )[0]
        _closest_match = _lower_case_to_original_possibility_dict[
            _lower_case_closest_match]
        return _closest_match
    except IndexError:
        _error_msg = \
            f'Cannot find \'{word}\' in all possibilities ' \
            f'({possibilities}) within the similarity cutoff.'
        raise ValueError(_error_msg)
