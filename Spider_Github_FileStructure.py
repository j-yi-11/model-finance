# -*- coding: utf-8 -*-
import time
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import warnings
"""避免给注释斜体和加粗"""
"""
@ ToDo:   Crawl the file structure of the GitHub project
@ author: pump
@ date: 2021/11/05
@ 适配性
    问题1，爬取 laravel 5.7
        正常的项目根目录路径是： https://github.com/laravel/laravel/tree/5.7，https://github.com/laravel/laravel/tree/8.x（示例中的是默认页）
        更正：项目根目录地址——修改，前缀和后缀——适配无需修改          
@ 思路：
    访问页面得到的原始数据：一堆超链接。
    筛选原始数据：根据路径。
    处理数据：是目录就下一层。是文件就保存路径。

    第二层，继续调用该方法。需要增加前缀以免重复爬取同一页面，但不能直接增加前缀，否则无法回溯到根目录继续爬取。

@ Github 项目路径信息：
    目录路径：https://github.com/laravel/laravel/tree/8.x/，目录都在/tree目录下
    文件路径：https://github.com/laravel/laravel/blob/8.x/，文件都在/blob/目录下
    补充：目前未确定，是否所有Github项目都以 /tree/ 和 /blob/ 路径来存储文件

@ 算法步骤：
    访问项目根目录页面，把超链接划分成三种：无关类、目录类、文件类
    @@@ 第一层
    （1）无关类：筛选掉
    （2）文件类：写入文件。依据：具有前缀/blob/8.x/
    （3）目录类：多层迭代访问。依据：具有前缀/tree/8.x/

    目录迭代访问算法：
        @@@ 第二层
            例如超链接的相对路径是：/laravel/laravel/tree/8.x/app，此时前缀是/laravel/laravel/tree/8.x/，
    需要对 url 、两个前缀进行更新以下一次调用。此时 url = urlRoot + href
        （1）更新 url：为了求 url ，要从超链接中截取出 "app"，取名后缀tail = href[len(directoryPre):] = "app"
        （2）更新目录前缀：directoryPre + tail
        （3）更新文件前缀：filePre + tail
        @@@ 错误更正1
            错误：不能直接修改前缀的值，这样会导致后续的前缀值无法还原，比如使用前缀/xx/app筛选项目根目录，结果将为空
            更正：传递参数时加上tail，不改变变量值

        @@@ 错误更正2
            错误：保存文件名时，没有添加目录前缀
            更正：定义一个初始前缀变量，把初始前缀变量删除即可

        @@@ 结果保存
            需求：每次启动脚本时，首先清空 .txt 文件内容。
            限制：不能把 write_path() 的文件打开方式改成 w ，因为每次递归都会打开一次
            解决：定义一个清空的函数，启动时写入""来清空

        @@@ 访问情况备注
            使用科学上网软件开启系统代理模式，访问Github几乎无失败记录，非常丝滑
"""

"""
函数说明
    class::__init__(urlRoot, url, dicFilename, directoryPre, filePre) 初始化变量/配置项

    class::write_path(filename, fileList) 把文件路径写入结果文件

    class::write_flush(filename) 清空存放文件路径的结果文件

    class::crawl_github_file_structure(urlRoot, url, dicFilename, directoryPre, filePre) 递归爬取目录、把文件路径写入结果文件

参数说明
    dicFilename = "projectStructure/laravel/laravel_8.x_fileStructure.txt"   变量：存放结果的文件
    urlRoot = "https://github.com"                  常量
    url = "https://github.com/laravel/laravel/"     变量：项目根目录地址
    directoryPre = "/tree/8.x/"                     变量：项目目录存储的路径
    filePre = "/blob/8.x/"                          变量：项目文件存储的路径

class使用说明
    crawl = CrawlGithub(urlRoot, url, dicFilename, directoryPre, filePre)

    crawl.write_flush(dicFilename)
    crawl.crawl_github_file_structure(urlRoot, url, dicFilename, directoryPre, filePre)

有效代码量：64 lines
实例结果：见文件末尾
"""


class CrawlGithub:

    def __init__(self, urlRoot, url, dicFilename, directoryPre, filePre):
        self.urlRoot = urlRoot
        self.url = url
        self.dicFilename = dicFilename
        self.directoryPre = directoryPre
        self.filePre = filePre
        self.initDirectoryPre = directoryPre
        self.initFilePre = filePre

    def write_path(self, filename, fileList):
        with open(filename, 'a+') as f:
            for file in fileList:
                f.write(file + "\n")
            f.close()

    def write_flush(self, filename):
        with open(filename, 'w') as f:
            f.write("")
            f.close()

    def crawl_github_file_structure(self, urlRoot, url, dicFilename, directoryPre, filePre):
        try:
            # 设置重连3次
            s = requests.session()
            # s.mount('http://', HTTPAdapter(max_retries=3))
            s.mount('https://', HTTPAdapter(max_retries=3))
            s.keep_alive = False
            resp = s.get(url, timeout=(20, 100))#, verify=False, headers={'Connection':'close'})
            # time.sleep(30)
            print("访问成功")

            fileList = []
            soup = BeautifulSoup(resp.text, "lxml")
            list_a = soup.find_all('a')
            for a in list_a:
                href = a.get('href')
                print("href = ", href)

                # 只做白名单即可：目录路径前缀 /tree/8.x/、文件路径前缀 /blob/8.x/
                if href is not None and directoryPre in href:  # 目录
                    url = urlRoot + href  # 新的访问页面
                    tail = href[16 + len(directoryPre):]
                    print(url)
                    self.crawl_github_file_structure(urlRoot, url, dicFilename, directoryPre + tail, filePre + tail)

                if href is not None and filePre in href:
                    href = href[href.find(self.initFilePre) + len(self.initFilePre):]
                    fileList.append(href)
            self.write_path(dicFilename, fileList)

        except requests.exceptions.ConnectTimeout:
            print("网络连接超时")
        except requests.exceptions.ReadTimeout:
            print("读取时间超时")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    dicFilename = "fileStructure.txt"
    urlRoot = "https://github.com"
    url = "https://github.com/j-yi-11/IMPUS-Modified-For-LocalRun"
    directoryPre = "/j-yi-11/IMPUS-Modified-For-LocalRun/tree"
    filePre = "/j-yi-11/IMPUS-Modified-For-LocalRun/blob"

    crawl = CrawlGithub(urlRoot, url, dicFilename, directoryPre, filePre)

    crawl.write_flush(dicFilename)
    crawl.crawl_github_file_structure(urlRoot, url, dicFilename, directoryPre, filePre)

'''
@@@ 结果示例
app/Console/Kernel.php
app/Exceptions/Handler.php
app/Http/Controllers/Controller.php
app/Http/Middleware/Authenticate.php
app/Http/Middleware/EncryptCookies.php
app/Http/Middleware/PreventRequestsDuringMaintenance.php
app/Http/Middleware/RedirectIfAuthenticated.php
app/Http/Middleware/TrimStrings.php
app/Http/Middleware/TrustHosts.php
app/Http/Middleware/TrustProxies.php
app/Http/Middleware/VerifyCsrfToken.php
app/Http/Kernel.php
app/Models/User.php
app/Providers/AppServiceProvider.php
app/Providers/AuthServiceProvider.php
app/Providers/BroadcastServiceProvider.php
app/Providers/EventServiceProvider.php
app/Providers/RouteServiceProvider.php
bootstrap/cache/.gitignore
bootstrap/app.php
config/app.php
config/auth.php
config/broadcasting.php
config/cache.php
config/cors.php
config/database.php
config/filesystems.php
config/hashing.php
config/logging.php
config/mail.php
config/queue.php
config/sanctum.php
config/services.php
config/session.php
config/view.php
database/factories/UserFactory.php
database/migrations/2014_10_12_000000_create_users_table.php
database/migrations/2014_10_12_100000_create_password_resets_table.php
database/migrations/2019_08_19_000000_create_failed_jobs_table.php
database/migrations/2019_12_14_000001_create_personal_access_tokens_table.php
database/seeders/DatabaseSeeder.php
database/.gitignore
public/.htaccess
public/favicon.ico
public/index.php
public/robots.txt
public/web.config
resources/css/app.css
resources/js/app.js
resources/js/bootstrap.js
resources/lang/en/auth.php
resources/lang/en/pagination.php
resources/lang/en/passwords.php
resources/lang/en/validation.php
resources/views/welcome.blade.php
routes/api.php
routes/channels.php
routes/console.php
routes/web.php
storage/app/public/.gitignore
storage/app/.gitignore
storage/framework/cache/data/.gitignore
storage/framework/cache/.gitignore
storage/framework/sessions/.gitignore
storage/framework/testing/.gitignore
storage/framework/views/.gitignore
storage/framework/.gitignore
storage/logs/.gitignore
tests/Feature/ExampleTest.php
tests/Unit/ExampleTest.php
tests/CreatesApplication.php
tests/TestCase.php
.editorconfig
.env.example
.gitattributes
.gitignore
.styleci.yml
CHANGELOG.md
README.md
artisan
composer.json
package.json
phpunit.xml
server.php
webpack.mix.js
'''

