def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.lr_decay = 100

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.lr_decay = 150

    if args.template.find('DMMFSR') >= 0:
        args.model = 'DMMFSR'
        args.block = 5
        args.n_layer = 8
        args.n_feats = 64
        args.kernel_size = 3
        args.rate = 64
        args.in_dim = 64
        args.chop = True

