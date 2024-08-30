import google_play_scraper
from google_play_scraper import Sort
from google_play_scraper.constants.element import ElementSpecs
from google_play_scraper.constants.regex import Regex
from google_play_scraper.constants.request import Formats
from google_play_scraper.utils.request import post

import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import json
from time import sleep
from typing import List, Optional, Tuple

reviews_count = 25000

MAX_COUNT_EACH_FETCH = 199


class _ContinuationToken:
    __slots__ = (
        "token",
        "lang",
        "country",
        "sort",
        "count",
        "filter_score_with",
        "filter_device_with",
    )

    def __init__(
        self, token, lang, country, sort, count, filter_score_with, filter_device_with
    ):
        self.token = token
        self.lang = lang
        self.country = country
        self.sort = sort
        self.count = count
        self.filter_score_with = filter_score_with
        self.filter_device_with = filter_device_with


def _fetch_review_items(
    url: str,
    app_id: str,
    sort: int,
    count: int,
    filter_score_with: Optional[int],
    filter_device_with: Optional[int],
    pagination_token: Optional[str],
):
    dom = post(
        url,
        Formats.Reviews.build_body(
            app_id,
            sort,
            count,
            "null" if filter_score_with is None else filter_score_with,
            "null" if filter_device_with is None else filter_device_with,
            pagination_token,
        ),
        {"content-type": "application/x-www-form-urlencoded"},
    )
    match = json.loads(Regex.REVIEWS.findall(dom)[0])

    return json.loads(match[0][2])[0], json.loads(match[0][2])[-2][-1]


def reviews(
    app_id: str,
    lang: str = "en",
    country: str = "us",
    sort: Sort = Sort.MOST_RELEVANT,
    count: int = 100,
    filter_score_with: int = None,
    filter_device_with: int = None,
    continuation_token: _ContinuationToken = None,
) -> Tuple[List[dict], _ContinuationToken]:
    sort = sort.value

    if continuation_token is not None:
        token = continuation_token.token

        if token is None:
            return (
                [],
                continuation_token,
            )

        lang = continuation_token.lang
        country = continuation_token.country
        sort = continuation_token.sort
        count = continuation_token.count
        filter_score_with = continuation_token.filter_score_with
        filter_device_with = continuation_token.filter_device_with
    else:
        token = None

    url = Formats.Reviews.build(lang=lang, country=country)

    _fetch_count = count

    result = []

    while True:
        if _fetch_count == 0:
            break

        if _fetch_count > MAX_COUNT_EACH_FETCH:
            _fetch_count = MAX_COUNT_EACH_FETCH

        try:
            review_items, token = _fetch_review_items(
                url,
                app_id,
                sort,
                _fetch_count,
                filter_score_with,
                filter_device_with,
                token,
            )
        except (TypeError, IndexError):
            #funnan MOD start
            token = continuation_token.token
            continue
            #MOD end

        for review in review_items:
            result.append(
                {
                    k: spec.extract_content(review)
                    for k, spec in ElementSpecs.Review.items()
                }
            )

        _fetch_count = count - len(result)

        if isinstance(token, list):
            token = None
            break

    return (
        result,
        _ContinuationToken(
            token, lang, country, sort, count, filter_score_with, filter_device_with
        ),
    )


def reviews_all(app_id: str, sleep_milliseconds: int = 0, **kwargs) -> list:
    kwargs.pop("count", None)
    kwargs.pop("continuation_token", None)

    continuation_token = None

    result = []

    while True:
        _result, continuation_token = reviews(
            app_id,
            count=MAX_COUNT_EACH_FETCH,
            continuation_token=continuation_token,
            **kwargs
        )

        result += _result

        if continuation_token.token is None:
            break

        if sleep_milliseconds:
            sleep(sleep_milliseconds / 1000)

    return result




def get_app_reviews_dataframe(
    app_id: str,
    reviews_count: int = 25000,
    lang: str = 'en',
    country: str = 'in',
    sort: Sort = Sort.NEWEST,
    filter_score_with: Optional[int] = None,
    filter_device_with: Optional[int] = None,
    sleep_milliseconds: int = 0
) -> pd.DataFrame:
    """
    Fetch app reviews and return them as a pandas DataFrame.

    :param app_id: The ID of the app to fetch reviews for.
    :param reviews_count: The number of reviews to fetch (default 25000).
    :param lang: The language of the reviews (default 'en').
    :param country: The country for which to fetch reviews (default 'in').
    :param sort: The sort order for reviews (default Sort.NEWEST).
    :param filter_score_with: Filter reviews by score (default None).
    :param filter_device_with: Filter reviews by device (default None).
    :param sleep_milliseconds: Sleep duration between requests in milliseconds (default 0).
    :return: A pandas DataFrame containing the app reviews.
    """
    result = []
    continuation_token = None

    with tqdm(total=reviews_count, position=0, leave=True) as pbar:
        while len(result) < reviews_count:
            new_result, continuation_token = reviews(
                app_id,
                continuation_token=continuation_token,
                lang=lang,
                country=country,
                sort=sort,
                filter_score_with=filter_score_with,
                filter_device_with=filter_device_with,
                count=MAX_COUNT_EACH_FETCH
            )
            if not new_result:
                break
            result.extend(new_result)
            pbar.update(len(new_result))

            if sleep_milliseconds:
                sleep(sleep_milliseconds / 1000)

    return pd.DataFrame(result)
