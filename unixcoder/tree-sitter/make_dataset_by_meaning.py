from tree_sitter import Language, Parser
import json


def mask_code(src_string, mask_which='function_name', mask_token='<mask0>'):
    # INPUT:
    #   src_string: code to be modified by replacing some token with the mask_token
    #   mask_which: "function_name" or "function_call"
    #   mask_token:
    Language.build_library(
        # Store the library in the `build` directory
        'build/my-languages.so',

        # Include one or more languages
        ['vendor/tree-sitter-python']
    )

    PY_LANGUAGE = Language('build/my-languages.so', 'python')

    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    src = bytes(src_string, "utf8")
    tree = parser.parse(src)

    if mask_which == "function_name":
        query = PY_LANGUAGE.query("""
        (function_definition
            name: (identifier) @function.def)
        """)
    elif mask_which == "function_call":
        query = PY_LANGUAGE.query("""
        (call
            function: (identifier) @function.call)
        """)

    gt = ''
    captures = query.captures(tree.root_node)
    if len(captures) > 0:
        start_point = captures[0][0].start_point
        end_point = captures[0][0].end_point

        src_lines = src_string.splitlines(True)
        assert start_point[0] == end_point[0]
        gt = src_lines[start_point[0]][start_point[1]:end_point[1]]

    return src_string.replace(gt, mask_token), gt


with open("../test1.txt", "r") as f:
    replace_to = '<mask0>'
    for line in f:
        line = line.replace("<EOL>", '\n')
        line = line.replace("<s>", '\n')
        line = line.replace("</s>", '')
        line, gt = mask_code(line)

        test_dict = {"input": line, "gt": gt}
        with open("test1.json", "a") as f2:
            if gt != '':
                f2.write(json.dumps(test_dict) + '\n')

