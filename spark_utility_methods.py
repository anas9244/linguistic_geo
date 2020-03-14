def all_recsys_twitter_splits():
    from os import listdir
    from os.path import isfile, join
    path_name = "/mnt/ceph/storage/data-in-progress/kibi9872/twitter-recsys-2020/splitted-data/"
    ret = (join(path_name, f) for f in listdir(path_name))

    return (f for f in ret if isfile(f))

def parse_recsys_file(file_name):
    from csv import DictReader
    import codecs
    delimiter = ''
    fieldnames=[#Tweet Features
            'tweet-text-tokens', 'tweet-hashtags', 'tweet-id',
            'tweet-present-media', 'tweet-present-links', 'tweet-present-domains',
            'tweet-type', 'tweet-language', 'tweet-timestamp',
            #Engaged With User Features
            'user-id', 'user-follower-count', 'user-following-count',
            'user-is-verified', 'user-account-creation-time',
            #Engaging User Features
            'engaging-user-id', 'engaging-user-follower-count', 'engaging-user-following-count',
            'engaging-user-is-verified', 'engaging-user-account-creation-time'
            #Engagement Features
            'engagee-follows-engager', 'reply-engagement-timestamp', 'retweet-engagement-timestamp',
            'retweet-with-comment-engagement-timestamp', 'like-engagement-timestamp'
            ]
    with open(file_name, 'r') as f:
        csv_reader = DictReader(f, delimiter=delimiter, fieldnames=fieldnames)
        return [i for i in csv_reader]

def tweet_rdd(sc):
    return sc.parallelize(['IGNORE-ME'])\
        .flatMap(lambda i: all_recsys_twitter_splits())\
        .flatMap(lambda i: parse_recsys_file(i))

