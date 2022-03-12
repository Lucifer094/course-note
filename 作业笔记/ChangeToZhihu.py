import re
import os
import sys
PATH_1 = '<img src="https://www.zhihu.com/equation?tex={}\\\\" alt="{}\\\\" class="ee_img tr_noresize" eeimg="1">'
PATH_2 = '<img src="https://www.zhihu.com/equation?tex={}" alt="{}" class="ee_img tr_noresize" eeimg="1">'


def replace_func(source_file: str, target_file: str, enable_tag=True):
    if not (source_file.endswith(".txt") or source_file.endswith(".md")):
        print("仅支持.md文件或.txt文件！")
        return False

    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()

    if (enable_tag == True):
        line_equations_format = re.compile(r'\$\$(.*?)\$\$', re.S)
        line_equations = re.findall(line_equations_format, content)
        for index, line_equation in enumerate(line_equations):
            tag = "\\tag{" + str(index+1) + "}" if enable_tag and "\\tag" not in line_equation else ""
            prev_eq = "$${}$$".format(line_equation)
            new_eq = "{}".format(PATH_1.format(line_equation+tag, line_equation+tag))
            content = content.replace(prev_eq, new_eq)
    elif (enable_tag == False):
        line_equations_format = re.compile(r'\$\$(.*?)\$\$', re.S)
        line_equations = re.findall(line_equations_format, content)
        for index, line_equation in enumerate(line_equations):
            prev_eq = "$${}$$".format(line_equation)
            new_eq = "{}".format(PATH_1.format(line_equation, line_equation))
            content = content.replace(prev_eq, new_eq)

    inline_equations_format = re.compile(r'\$(.*?)\$', re.S)
    inline_equations = re.findall(inline_equations_format, content)
    for index, inline_equation in enumerate(inline_equations):
        prev_eq = "${}$".format(inline_equation)
        new_eq = "{}".format(PATH_2.format(inline_equation, inline_equation))
        content = content.replace(prev_eq, new_eq)

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)


def main(enable_tag=True):
    path = sys.path[0]  # 获取当前运行路径
    filename_list = os.listdir(path)  # 获取所有文件名的列表
    originname_list = [filename for filename in filename_list
                        if filename.endswith((".md"))]  # 获取所有Markdown文件名列表
    for originname in originname_list:  # 分离.md文件名称和后缀，转化为带有_for_zhihu的名称
        zhihuname = os.path.splitext(originname)[0] + '_for_zhihu' + '.md'
        if '_for_zhihu' in os.path.splitext(originname)[0]:
            continue  # 如果当前.md文件对应的_for_zhihu版文件存在，则不执行转化操作
        origin = os.path.join(path, originname)  # 拼接路径和文件名
        zhihupath = os.path.join(path, zhihuname)
        replace_func(origin, zhihupath, enable_tag)


if __name__ == "__main__":
    main(True) # True, 带标号的行间公式; False: 不带标号的行间公式