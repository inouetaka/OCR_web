import toml


def config(root="./config/option.toml"):
    opt = toml.load(open(root))
    char_root = opt['opt']['character']
    with open(char_root, mode='r') as char:
        character = char.read()
    opt['opt']['character'] = character

    return opt['opt']