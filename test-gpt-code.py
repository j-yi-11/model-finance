import json
from urllib.request import Request, urlopen

class CodeRepository:
    def __init__(self):
        self.user = ""        # 用户信息
        self.repos = []       # 仓库列表
        self.file_tree = {}   # 文件结构
        self.metadata = {}    # 提交历史/star数等


def fetch_data(url, headers):
    req = Request(url, headers=headers)
    response = urlopen(req).read()
    return json.loads(response.decode())


def get_branches(owner, repo_name, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/branches'
    return fetch_data(url, headers)


def get_file_tree(owner, repo_name, branch, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1'
    return fetch_data(url, headers)


def get_repo_info(owner, repo_name, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}'
    return fetch_data(url, headers)


def get_issues(owner, repo_name, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/issues'
    return fetch_data(url, headers)


def get_issue_comments(owner, repo_name, issue_number, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/issues/{issue_number}/comments'
    return fetch_data(url, headers)


def get_collaborators(owner, repo_name, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/contributors'
    try:
        return fetch_data(url, headers)
    except Exception as e:
        print(f"Failed to fetch contributors: {e}")
        return []


def get_commits(owner, repo_name, headers):
    url = f'https://api.github.com/repos/{owner}/{repo_name}/commits'
    return fetch_data(url, headers)


if __name__ == '__main__':
    owner = 'kakao'
    repo_name = 'adt'
    headers = {'User-Agent': 'Mozilla/5.0',
               'Authorization': 'token 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',
               'Content-Type': 'application/json',
               'Accept': 'application/json'}

    repo = CodeRepository()
    repo.user = owner

    # 获取基本仓库信息
    repo_info = get_repo_info(owner, repo_name, headers)
    repo.repos.append(repo_info['name'])
    repo.metadata['stars'] = repo_info['stargazers_count']
    repo.metadata['description'] = repo_info['description']
    repo.metadata['open_issues'] = repo_info['open_issues_count']

    # 获取所有分支
    branches = get_branches(owner, repo_name, headers)
    for branch in branches:
        branch_name = branch['name']
        file_tree = get_file_tree(owner, repo_name, branch_name, headers)
        repo.file_tree[branch_name] = file_tree

    # 获取协作者
    repo.metadata['collaborators'] = [collab['login'] for collab in get_collaborators(owner, repo_name, headers)]

    # 获取 commits 历史
    repo.metadata['commits'] = get_commits(owner, repo_name, headers)

    # 获取 issues 及其评论
    issues = get_issues(owner, repo_name, headers)
    repo.metadata['issues'] = []
    for issue in issues:
        issue_data = {
            'title': issue['title'],
            'body': issue['body'],
            'comments': []
        }
        if issue['comments'] > 0:
            comments = get_issue_comments(owner, repo_name, issue['number'], headers)
            issue_data['comments'] = [comment['body'] for comment in comments]
        repo.metadata['issues'].append(issue_data)

    # 保存最终结果
    with open(f"./{owner}-{repo_name}-full-data.json", "w", encoding="utf-8") as f:
        json.dump(repo.__dict__, f, indent=4)