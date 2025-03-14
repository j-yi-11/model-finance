import time
from urllib.request import urlopen
from urllib.request import Request
import json

# reference:
# https://blog.csdn.net/weixin_39132520/article/details/114925354
# https://blog.csdn.net/soldi_er/article/details/121138032

def get_results(search, headers, page, stars):
    # url = 'https://api.github.com/search/repositories?q={search}%20stars:<={stars}&page={num}&per_page=100&sort=stars' \
    #       '&order=desc'.format(search=search, num=page, stars=stars)
    # url = 'https://api.github.com/repos/{owner}'.format(owner='kakao')#, repo='ZJU_CS_Related_Course') /{repo}
    # url = 'https://api.github.com/search/repositories?q={query}'.format(query='owner:j-yi-11')
    # url = 'https://api.github.com/users/{owner}/repos'.format(owner='kakao')
    # url = 'https://api.github.com/repos/kakao/{repo_name}/branches'.format(repo_name='awesome-tech-newletters')
    url = 'https://api.github.com/repos/kakao/{repo_name}/git/trees/master?recursive=1'.format(repo_name='awesome-tech-newletters')
    req = Request(url, headers=headers)
    response = urlopen(req).read()
    result = json.loads(response.decode())
    return result


if __name__ == '__main__':
    # Specify JavaScript Repository
    search = 'language:python'

    # Modify the GitHub token value
    headers = {'User-Agent': 'Mozilla/5.0',
               'Authorization': 'token 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',
               'Content-Type': 'application/json',
               'Accept': 'application/json'
               }

    count = 1

    results = get_results(search, headers, 1, 321701)

    # write results to json
    with open("./kakao-adt-file-tree.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # stars = 421701
    # for i in range(0, 2):
    #     repos_list = []
    #     stars_list = []
    #     for page in range(1, 11):
    #         results = get_results(search, headers, page, stars)
    #         for item in results['items']:
    #             repos_list.append([count, item["name"], item["clone_url"]])
    #             stars_list.append(item["stargazers_count"])
    #             count += 1
    #         # print(len(repos_list))
    #     stars = stars_list[-1]
    #     # print(stars)
    #     # with open("./top2000Repos.txt", "a", encoding="utf-8") as f:
    #     #     for i in range(len(repos_list)):
    #     #         f.write(str(repos_list[i][0]) + "," + repos_list[i][1] + "," + repos_list[i][2] + "\n")
    #     # For authenticated requests, 30 requests per minute
    #     # For unauthenticated requests, the rate limit allows you to make up to 10 requests per minute.
    #     time.sleep(60)
