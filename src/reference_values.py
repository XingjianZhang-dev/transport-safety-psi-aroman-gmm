"""Published values of the article, used as calibration targets and for figure ordering.

Country order follows Table 1, which is also the axis order of the Fig. 7 radar charts and
therefore the row order of the decision matrices.
"""
import pandas as pd

C = ["AR", "AU", "BR", "CA", "CH", "FR", "DE", "IN", "ID", "IT",
     "JP", "KR", "MX", "RU", "SA", "ZA", "TR", "GB", "US"]
NAMES = ["Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany", "India",
         "Indonesia", "Italy", "Japan", "South Korea", "Mexico", "Russia", "Saudi Arabia",
         "South Africa", "Turkey", "United Kingdom", "United States"]
YEARS = [2010, 2015, 2019, 2023]
INDICATORS = ["A31", "A32", "A33", "B21", "B22", "C41", "C42", "C43", "C44"]
INDICATOR_NAMES = {
    "A31": "Road fatalities per 100,000 inhabitants",
    "A32": "Road fatalities per 10,000 registered vehicles",
    "A33": "Change in the number of road deaths (%)",
    "B21": "Seatbelt wearing rates in front seats (%)",
    "B22": "Seatbelt wearing rates in rear seats (%)",
    "C41": "Enforcement score on speed limit law",
    "C42": "Enforcement score on drink-driving law",
    "C43": "Enforcement score on seat-belt law",
    "C44": "Enforcement score on helmet use law",
}
COST = [0, 1, 2]                      # A31, A32, A33 are cost criteria

# Table 1 - composite scores and rankings
T1_SCORE = pd.DataFrame({
    2010: [.076, .165, .106, .162, .120, .176, .173, .068, .134, .155, .156, .142, .101, .117, .083, .044, .155, .180, .152],
    2015: [.120, .207, .109, .201, .098, .208, .198, .056, .138, .165, .210, .150, .100, .111, .074, .041, .097, .205, .160],
    2019: [.104, .178, .120, .167, .104, .178, .176, .033, .157, .148, .179, .155, .123, .107, .065, .069, .134, .190, .150],
    2023: [.096, .208, .093, .194, .093, .198, .210, .068, .114, .168, .228, .118, .079, .090, .086, .053, .085, .242, .140],
}, index=C)
T1_RANK = pd.DataFrame({
    2010: [17, 4, 14, 5, 12, 2, 3, 18, 11, 7, 6, 10, 15, 13, 16, 19, 8, 1, 9],
    2015: [11, 3, 13, 5, 15, 2, 6, 18, 10, 7, 1, 9, 14, 12, 17, 19, 16, 4, 8],
    2019: [15, 3, 13, 6, 16, 4, 5, 19, 7, 10, 2, 8, 12, 14, 18, 17, 11, 1, 9],
    2023: [11, 4, 12, 6, 13, 5, 3, 18, 10, 7, 2, 9, 17, 14, 15, 19, 16, 1, 8],
}, index=C)

# Table 2 - group membership
_T2 = {"AR": [4, 4, 3, 3], "AU": [1, 1, 1, 1], "BR": [4, 3, 2, 1], "CA": [1, 1, 1, 1],
       "CH": [4, 2, 4, 1], "DE": [1, 1, 1, 1], "FR": [1, 1, 1, 1], "GB": [1, 1, 1, 1],
       "ID": [1, 1, 1, 3], "IN": [3, 2, 4, 4], "IT": [2, 3, 3, 2], "JP": [2, 1, 2, 2],
       "KR": [2, 3, 2, 2], "MX": [3, 3, 2, 3], "RU": [4, 3, 3, 3], "SA": [3, 4, 2, 1],
       "TR": [1, 2, 4, 3], "US": [1, 1, 1, 1], "ZA": [2, 3, 3, 4]}
T2_GROUP = pd.DataFrame(_T2, index=YEARS).T.loc[C]
