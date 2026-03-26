"""
Embedded sample data for Campaign RAG Agent.
Contains both conversion and channel engagement datasets.
"""

CONVERSIONS_CSV = """campaign_id,campaign_name,treatment,segment,conversions,impressions,clicks,conversion_rate,cost_per_conversion
CAMP001,Summer Launch,Control,New Users,862,18795,2698,0.0459,6.79
CAMP002,Retention Boost,Variant A,Existing Users,43,9265,373,0.0046,8.93
CAMP003,Awareness Blast,Variant B,High Value,163,14363,554,0.0113,2.45
CAMP004,Holiday Promo,Optimized,All Users,36,5433,237,0.0066,4.43
CAMP005,New Product Launch,Control,New Users,133,9396,715,0.0142,6.89
CAMP001,Summer Launch,Variant A,Existing Users,91,5747,388,0.0158,8.28
CAMP002,Retention Boost,Variant B,High Value,118,4899,425,0.0241,2.37
CAMP003,Awareness Blast,Optimized,All Users,175,6890,747,0.0254,2.11
CAMP004,Holiday Promo,Control,New Users,587,11792,1716,0.0498,4.44
CAMP005,New Product Launch,Variant A,Existing Users,48,5612,280,0.0086,7.47
CAMP001,Summer Launch,Variant B,High Value,174,18787,1585,0.0093,9.27
CAMP002,Retention Boost,Optimized,All Users,94,4585,486,0.0205,6.16
CAMP003,Awareness Blast,Control,New Users,168,16417,1539,0.0102,8.74
CAMP004,Holiday Promo,Variant A,Existing Users,454,8675,1233,0.0523,6.78
CAMP005,New Product Launch,Variant B,High Value,72,14534,457,0.0050,2.36
CAMP001,Summer Launch,Optimized,All Users,286,11529,1039,0.0248,9.72
CAMP002,Retention Boost,Control,New Users,120,9331,844,0.0129,8.42
CAMP003,Awareness Blast,Variant A,Existing Users,97,19448,428,0.0050,5.16
CAMP004,Holiday Promo,Variant B,High Value,40,5695,117,0.0070,7.65
CAMP005,New Product Launch,Optimized,All Users,587,16986,2084,0.0346,9.41
CAMP001,Summer Launch,Control,New Users,452,9184,1276,0.0492,5.60
CAMP002,Retention Boost,Variant A,Existing Users,174,5454,583,0.0319,6.73
CAMP003,Awareness Blast,Variant B,High Value,456,13965,1889,0.0327,2.96
CAMP004,Holiday Promo,Optimized,All Users,691,16773,2186,0.0412,3.89
CAMP005,New Product Launch,Control,New Users,339,15685,1320,0.0216,5.42
CAMP001,Summer Launch,Variant A,Existing Users,626,18934,2583,0.0331,6.51
CAMP002,Retention Boost,Variant B,High Value,208,17820,933,0.0117,8.04
CAMP003,Awareness Blast,Optimized,All Users,161,4059,578,0.0397,7.56
CAMP004,Holiday Promo,Control,New Users,263,13817,1397,0.0190,2.84
CAMP005,New Product Launch,Variant A,Existing Users,275,16931,749,0.0162,6.31
CAMP001,Summer Launch,Variant B,High Value,352,13230,1805,0.0266,2.88
CAMP002,Retention Boost,Optimized,All Users,112,4306,325,0.0260,8.89
CAMP003,Awareness Blast,Control,New Users,113,10526,435,0.0107,5.88
CAMP004,Holiday Promo,Variant A,Existing Users,404,16545,1057,0.0244,4.59
CAMP005,New Product Launch,Variant B,High Value,54,6304,457,0.0086,4.03
CAMP001,Summer Launch,Optimized,All Users,238,14649,1649,0.0162,9.98
CAMP002,Retention Boost,Control,New Users,62,7737,568,0.0080,4.76
CAMP003,Awareness Blast,Variant A,Existing Users,162,16949,658,0.0096,9.89
CAMP004,Holiday Promo,Variant B,High Value,50,7931,241,0.0063,3.94
CAMP005,New Product Launch,Optimized,All Users,145,7389,501,0.0196,7.07
CAMP001,Summer Launch,Control,New Users,210,4930,621,0.0426,3.21
CAMP002,Retention Boost,Variant A,Existing Users,575,14589,1611,0.0394,4.61
CAMP003,Awareness Blast,Variant B,High Value,96,17243,382,0.0056,3.81
CAMP004,Holiday Promo,Optimized,All Users,73,5911,192,0.0123,5.18
CAMP005,New Product Launch,Control,New Users,210,5385,694,0.0390,7.88
CAMP001,Summer Launch,Variant A,Existing Users,878,17257,2419,0.0509,4.06
CAMP002,Retention Boost,Variant B,High Value,287,17075,729,0.0168,6.13
CAMP003,Awareness Blast,Optimized,All Users,106,16168,831,0.0066,9.18
CAMP004,Holiday Promo,Control,New Users,133,10158,650,0.0131,7.81
CAMP005,New Product Launch,Variant A,Existing Users,120,4154,545,0.0289,9.10
CAMP001,Summer Launch,Variant B,High Value,754,15874,2248,0.0475,7.35
CAMP002,Retention Boost,Optimized,All Users,485,18586,1271,0.0261,9.79
CAMP003,Awareness Blast,Control,New Users,64,5961,243,0.0107,7.54
CAMP004,Holiday Promo,Variant A,Existing Users,76,16992,728,0.0045,5.95
CAMP005,New Product Launch,Variant B,High Value,152,6987,472,0.0218,7.77
CAMP001,Summer Launch,Optimized,All Users,93,7735,726,0.0120,4.94
CAMP002,Retention Boost,Control,New Users,376,9893,956,0.0380,5.89
CAMP003,Awareness Blast,Variant A,Existing Users,306,8895,907,0.0344,6.02
CAMP004,Holiday Promo,Variant B,High Value,133,10022,842,0.0133,7.78
CAMP005,New Product Launch,Optimized,All Users,497,15946,1504,0.0312,2.35
CAMP001,Summer Launch,Control,New Users,281,13716,772,0.0205,7.98
CAMP002,Retention Boost,Variant A,Existing Users,130,4060,571,0.0320,9.73
CAMP003,Awareness Blast,Variant B,High Value,91,3699,484,0.0246,5.08
CAMP004,Holiday Promo,Optimized,All Users,122,9102,499,0.0134,4.98
CAMP005,New Product Launch,Control,New Users,60,3569,502,0.0168,3.67
CAMP001,Summer Launch,Variant A,Existing Users,148,7014,1043,0.0211,6.15
CAMP002,Retention Boost,Variant B,High Value,96,4409,273,0.0218,3.09
CAMP003,Awareness Blast,Optimized,All Users,194,15533,1036,0.0125,8.47
CAMP004,Holiday Promo,Control,New Users,135,8986,479,0.0150,2.65
CAMP005,New Product Launch,Variant A,Existing Users,559,15323,1896,0.0365,7.62
CAMP001,Summer Launch,Variant B,High Value,72,7495,199,0.0096,2.22
CAMP002,Retention Boost,Optimized,All Users,374,16121,1241,0.0232,4.63
CAMP003,Awareness Blast,Control,New Users,1036,19958,2946,0.0519,8.88
CAMP004,Holiday Promo,Variant A,Existing Users,61,12907,323,0.0047,6.30
CAMP005,New Product Launch,Variant B,High Value,424,18328,2339,0.0231,9.72
CAMP001,Summer Launch,Optimized,All Users,39,3876,175,0.0101,7.60
CAMP002,Retention Boost,Control,New Users,452,10079,1152,0.0448,6.13
CAMP003,Awareness Blast,Variant A,Existing Users,144,11311,624,0.0127,2.63
CAMP004,Holiday Promo,Variant B,High Value,78,11308,345,0.0069,7.51
CAMP005,New Product Launch,Optimized,All Users,131,4081,567,0.0321,3.92
CAMP001,Summer Launch,Control,New Users,276,8237,929,0.0335,4.24
CAMP002,Retention Boost,Variant A,Existing Users,164,6343,552,0.0259,5.49
CAMP003,Awareness Blast,Variant B,High Value,120,11130,581,0.0108,8.06
CAMP004,Holiday Promo,Optimized,All Users,36,12435,323,0.0029,8.84
CAMP005,New Product Launch,Control,New Users,114,14151,462,0.0081,5.79
CAMP001,Summer Launch,Variant A,Existing Users,152,7452,535,0.0204,7.08
CAMP002,Retention Boost,Variant B,High Value,32,5849,315,0.0055,9.47
CAMP003,Awareness Blast,Optimized,All Users,501,12823,1684,0.0391,3.30
CAMP004,Holiday Promo,Control,New Users,456,8878,1267,0.0514,7.09
CAMP005,New Product Launch,Variant A,Existing Users,79,3851,365,0.0205,7.15
CAMP001,Summer Launch,Variant B,High Value,303,7000,878,0.0433,7.01
CAMP002,Retention Boost,Optimized,All Users,144,8536,893,0.0169,4.19
CAMP003,Awareness Blast,Control,New Users,67,8726,602,0.0077,6.95
CAMP004,Holiday Promo,Variant A,Existing Users,423,19281,2094,0.0219,4.09
CAMP005,New Product Launch,Variant B,High Value,140,6267,689,0.0223,9.49
CAMP001,Summer Launch,Optimized,All Users,415,10543,1218,0.0394,4.26
CAMP002,Retention Boost,Control,New Users,297,14657,1100,0.0203,6.61
CAMP003,Awareness Blast,Variant A,Existing Users,115,8073,423,0.0142,8.94
CAMP004,Holiday Promo,Variant B,High Value,370,15043,1139,0.0246,8.04
CAMP005,New Product Launch,Optimized,All Users,369,16584,1483,0.0223,5.12"""

ENGAGEMENT_CSV = """channel,campaign_id,campaign_name,spend,impressions,clicks,ctr,avg_session_duration,engagement_score
Email,CAMP001,Summer Launch,1033.38,44954,2374,0.0528,199,0.979
Paid Social,CAMP002,Retention Boost,2572.00,9360,422,0.0451,187,0.741
Search Ads,CAMP003,Awareness Blast,1580.47,13335,791,0.0593,158,0.801
Organic Social,CAMP004,Holiday Promo,1750.37,26972,2501,0.0927,210,0.891
SMS,CAMP005,New Product Launch,517.70,29244,1882,0.0644,226,0.998
Display,CAMP001,Summer Launch,356.44,28714,3090,0.1076,134,0.559
Email,CAMP002,Retention Boost,600.38,28625,2921,0.1020,204,0.551
Paid Social,CAMP003,Awareness Blast,435.50,44081,2364,0.0536,86,0.853
Search Ads,CAMP004,Holiday Promo,427.78,31962,2363,0.0739,168,0.685
Organic Social,CAMP005,New Product Launch,2475.84,45262,5361,0.1184,191,0.675
SMS,CAMP001,Summer Launch,2182.91,39578,3779,0.0955,226,0.712
Display,CAMP002,Retention Boost,2737.79,33710,1948,0.0578,232,0.506
Email,CAMP003,Awareness Blast,1512.25,35355,815,0.0231,120,0.911
Paid Social,CAMP004,Holiday Promo,2283.79,44298,3285,0.0742,172,0.541
Search Ads,CAMP005,New Product Launch,1679.44,7105,80,0.0113,219,0.778
Organic Social,CAMP001,Summer Launch,1103.52,25559,1739,0.0680,223,0.777
SMS,CAMP002,Retention Boost,2914.05,11190,886,0.0792,73,0.647
Display,CAMP003,Awareness Blast,2988.33,49597,3683,0.0743,203,0.869
Email,CAMP004,Holiday Promo,2762.71,39121,4481,0.1145,161,0.728
Paid Social,CAMP005,New Product Launch,1936.37,34703,994,0.0286,223,0.732
Search Ads,CAMP001,Summer Launch,1189.39,36921,1449,0.0392,160,0.987
Organic Social,CAMP002,Retention Boost,2961.39,45080,1838,0.0408,216,0.567
SMS,CAMP003,Awareness Blast,413.36,10713,598,0.0558,106,0.960
Display,CAMP004,Holiday Promo,431.70,28793,2034,0.0706,64,0.513
Email,CAMP005,New Product Launch,1254.10,32712,2055,0.0628,236,0.688
Paid Social,CAMP001,Summer Launch,2298.82,25309,2561,0.1012,236,0.919
Search Ads,CAMP002,Retention Boost,1512.34,45034,1113,0.0247,208,0.528
Organic Social,CAMP003,Awareness Blast,2621.22,23384,1608,0.0688,192,0.998
SMS,CAMP004,Holiday Promo,1755.21,28524,2563,0.0899,142,0.624
Display,CAMP005,New Product Launch,1461.52,43765,1648,0.0377,232,0.803
Email,CAMP001,Summer Launch,840.20,42744,3986,0.0933,93,0.679
Paid Social,CAMP002,Retention Boost,517.96,19489,1262,0.0648,177,0.916
Search Ads,CAMP003,Awareness Blast,1061.78,47944,3390,0.0707,192,0.544
Organic Social,CAMP004,Holiday Promo,2417.09,27928,690,0.0247,106,0.710
SMS,CAMP005,New Product Launch,2397.07,17173,1501,0.0874,238,0.952
Display,CAMP001,Summer Launch,1928.34,44649,2168,0.0486,157,0.818
Email,CAMP002,Retention Boost,1753.48,15209,734,0.0483,225,0.725
Paid Social,CAMP003,Awareness Blast,2749.32,9637,755,0.0783,110,0.849
Search Ads,CAMP004,Holiday Promo,2430.12,24856,1594,0.0641,143,0.730
Organic Social,CAMP005,New Product Launch,2944.09,19388,934,0.0482,185,0.817
SMS,CAMP001,Summer Launch,872.41,11254,272,0.0242,139,0.576
Display,CAMP002,Retention Boost,588.72,29071,866,0.0298,191,0.673
Email,CAMP003,Awareness Blast,2711.01,41034,2735,0.0667,219,0.586
Paid Social,CAMP004,Holiday Promo,738.41,28954,2906,0.1004,151,0.674
Search Ads,CAMP005,New Product Launch,289.05,44749,1041,0.0233,71,0.603
Organic Social,CAMP001,Summer Launch,1219.96,31329,2692,0.0859,132,0.546
SMS,CAMP002,Retention Boost,1547.21,48221,915,0.0190,153,0.578
Display,CAMP003,Awareness Blast,1256.40,9432,381,0.0404,60,0.542
Email,CAMP004,Holiday Promo,1872.38,27195,898,0.0330,186,0.962
Paid Social,CAMP005,New Product Launch,861.13,33699,1704,0.0506,189,0.874
Search Ads,CAMP001,Summer Launch,302.71,10126,866,0.0855,210,0.756
Organic Social,CAMP002,Retention Boost,1689.92,45940,4841,0.1054,97,0.766
SMS,CAMP003,Awareness Blast,878.92,13152,1198,0.0911,87,0.510
Display,CAMP004,Holiday Promo,1101.82,34926,691,0.0198,185,0.560
Email,CAMP005,New Product Launch,2693.48,49243,3059,0.0621,197,0.895
Paid Social,CAMP001,Summer Launch,1595.64,31734,2356,0.0742,93,0.779
Search Ads,CAMP002,Retention Boost,1842.30,28275,679,0.0240,228,0.669
Organic Social,CAMP003,Awareness Blast,2719.01,39529,1457,0.0369,118,0.993
SMS,CAMP004,Holiday Promo,1896.17,32285,1123,0.0348,129,0.623
Display,CAMP005,New Product Launch,649.91,15589,453,0.0291,114,0.921
Email,CAMP001,Summer Launch,344.58,40358,2225,0.0551,95,0.999
Paid Social,CAMP002,Retention Boost,2710.51,46427,5415,0.1166,63,0.503
Search Ads,CAMP003,Awareness Blast,2930.19,23935,2142,0.0895,121,0.965
Organic Social,CAMP004,Holiday Promo,1758.94,18703,1167,0.0624,232,0.885
SMS,CAMP005,New Product Launch,723.72,47101,4154,0.0882,79,0.621
Display,CAMP001,Summer Launch,521.54,44001,4156,0.0945,97,0.791
Email,CAMP002,Retention Boost,632.22,44756,3069,0.0686,220,0.685
Paid Social,CAMP003,Awareness Blast,1471.95,21644,367,0.0170,231,0.601
Search Ads,CAMP004,Holiday Promo,2116.80,39387,1497,0.0380,160,0.501
Organic Social,CAMP005,New Product Launch,526.75,32534,2012,0.0618,222,0.674
SMS,CAMP001,Summer Launch,2802.68,36037,3317,0.0920,237,0.562
Display,CAMP002,Retention Boost,2246.43,10256,333,0.0325,94,0.963
Email,CAMP003,Awareness Blast,2220.47,10951,1050,0.0959,70,0.898
Paid Social,CAMP004,Holiday Promo,764.56,28325,2854,0.1008,224,0.907
Search Ads,CAMP005,New Product Launch,2062.55,45345,4981,0.1098,148,0.939
Organic Social,CAMP001,Summer Launch,1298.85,48303,2884,0.0597,163,0.688
SMS,CAMP002,Retention Boost,1495.50,11565,802,0.0693,122,0.751
Display,CAMP003,Awareness Blast,850.20,39016,2070,0.0531,112,0.953
Email,CAMP004,Holiday Promo,1947.87,6970,400,0.0574,126,0.814
Paid Social,CAMP005,New Product Launch,1137.74,26951,763,0.0283,170,0.810
Search Ads,CAMP001,Summer Launch,1693.69,14446,1055,0.0730,216,0.576
Organic Social,CAMP002,Retention Boost,1072.82,37154,3412,0.0918,145,0.857
SMS,CAMP003,Awareness Blast,1583.15,48872,1041,0.0213,200,0.555
Display,CAMP004,Holiday Promo,2570.07,11359,868,0.0764,74,0.899
Email,CAMP005,New Product Launch,619.77,19382,1065,0.0549,232,0.860
Paid Social,CAMP001,Summer Launch,1995.21,44666,3225,0.0722,85,0.626
Search Ads,CAMP002,Retention Boost,1167.95,17298,1997,0.1154,163,0.792
Organic Social,CAMP003,Awareness Blast,1322.38,5814,242,0.0416,166,0.577
SMS,CAMP004,Holiday Promo,1841.44,49241,3804,0.0773,170,0.863
Display,CAMP005,New Product Launch,1860.30,44915,3241,0.0722,127,0.895
Email,CAMP001,Summer Launch,264.46,5126,137,0.0267,151,0.687
Paid Social,CAMP002,Retention Boost,392.98,25530,2323,0.0910,99,0.920
Search Ads,CAMP003,Awareness Blast,2749.92,39560,888,0.0224,208,0.583
Organic Social,CAMP004,Holiday Promo,721.70,24256,1128,0.0465,114,0.614
SMS,CAMP005,New Product Launch,1902.10,16694,267,0.0160,196,0.603
Display,CAMP001,Summer Launch,2405.81,37409,1746,0.0467,122,0.707
Email,CAMP002,Retention Boost,2617.85,12973,623,0.0480,221,0.959
Paid Social,CAMP003,Awareness Blast,1843.79,46914,2803,0.0597,238,0.624
Search Ads,CAMP004,Holiday Promo,1817.37,10299,577,0.0560,147,0.656
Organic Social,CAMP005,New Product Launch,2385.39,9757,333,0.0341,173,0.832"""


def get_sample_dataframes():
    """Parse embedded CSV strings into pandas DataFrames."""
    import pandas as pd
    import io
    df_conv = pd.read_csv(io.StringIO(CONVERSIONS_CSV))
    df_eng = pd.read_csv(io.StringIO(ENGAGEMENT_CSV))
    return df_conv, df_eng
