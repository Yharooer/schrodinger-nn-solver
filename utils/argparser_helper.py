def reduce_to_single_argument(args, name):
    if len(args[name]) == 1:
        args[name] = args[name][0]
    else:
        raise ValueError(f'{name} requires only one argument, not {len(args[name])}.')


def reduce_to_single_arguments(args, names=None):
    if names == None:
        names = args.keys()

    for name in names:
        if isinstance(args[name], list):
            reduce_to_single_argument(args, name)


def check_arg_position_number(args, name):
    if args[name] <= 0:
        raise ValueError(f'{name} must be greater than zero.')


def check_args_positive_numbers(args, names=None):
    if names == None:
        names = args.keys()

    for name in names:
        if not name in args:
            raise ValueError(f'You asked me to check for \'{name}\' in the list of arguments but it wasn\'t there.')

        check_arg_position_number(args, name)


def print_arguments(args):
    print('Using the following parameters:')
    for k in args.keys():
        print(f'\t{k}: {args[k]}')
    print('\n')