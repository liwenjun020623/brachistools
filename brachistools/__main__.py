
from tqdm import tqdm
import numpy as np
from skimage import img_as_ubyte

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import os, sys
import csv

try:
    from brachistools.gui import gui
    GUI_ENABLED = True
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = False
    raise

import logging

from brachistools.segmentation import segmentation_pipeline, default_segmentation_params
from brachistools.classification import classification_pipeline
from brachistools.io import load_folder, imread, imsave, labels_to_xml, xml_to_labels
from brachistools.version import version_str

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Brachistools Command Line Parameters")
    parser.add_argument('--verbose', action='store_true', help="print additional messages")


    common_parser = argparse.ArgumentParser(add_help=False)
    input_img_args = common_parser.add_argument_group("Input Image Arguments")
    input_img_args.add_argument('--dir', default=[], type=str, help='folder containing data to run on')
    input_img_args.add_argument('--image_path', default=[], type=str,
                                help='run on single image')
    input_img_args.add_argument('--ignored_suffixes', default=['_mask', '_xml2seg'],
                                type=str, nargs='+', required=False,
                                help='ignore file if its root name ends with these suffixes')
    output_args = common_parser.add_argument_group("Output Arguments")
    output_args.add_argument('--save_format', required=False, default='PNG', type=str,
                             help="the file extension (no dot) of saved binary masks. Default is 'PNG'")
    output_args.add_argument('--save_dir', default=None, type=str,
                             help="folder to which segmentation results will be saved (defaults to input image directory)")
    output_args.add_argument('--save_npy', action='store_true',
                             help="save instance segmentation results as '.npy' labeled mask arrays. "
                             "For instance segmentation, the XML format will always be saved regardless of "
                             "this option")

    subparsers = parser.add_subparsers(dest='command')

    segment_subparser = subparsers.add_parser('segment', parents=[common_parser], help="Perform segmentation")
    classify_subparser = subparsers.add_parser('classify', parents=[common_parser], help="Perform classification (suggest diagnosis)")
    config_subparser = subparsers.add_parser('config', help="Edit program configurations")
    show_subparser = subparsers.add_parser('show', parents=[common_parser], help="Generate labels for segmentation XML")
    gui_subparser = subparsers.add_parser('gui', help="Launch GUI application")
    version_subparser = subparsers.add_parser('version', help="Show version info")

    segmentation_args = segment_subparser.add_argument_group("Segmentation Pipeline Arguments")
    segmentation_args.add_argument('--vahadane-sparsity_regularizer',
                                   required=False, type=float,
                                   default=default_segmentation_params['vahadane']['sparsity_regularizer'],
                                   help="sparsity regularizer of dictionary learning in Vahadane's "
                                   "H&E deconvolution algorithm. Smaller values lead to less learning "
                                   "capability and usually result in more complete nucleus shapes (increases "
                                   "false positive rate)")
    segmentation_args.add_argument('--equalize_adapthist-clip_limit',
                                   required=False, type=float,
                                   default=default_segmentation_params['equalize_adapthist']['clip_limit'],
                                   help="clip_limit parameter in skimage.exposure.equalize_adapthist")
    segmentation_args.add_argument('--small_objects-min_size',
                                   required=False, type=int,
                                   default=default_segmentation_params['remove_small_objects']['min_size'],
                                   help="minimum threshold size of connected regions of 1")
    segmentation_args.add_argument('--small_holes-area_threshold',
                                   required=False, type=int,
                                   default=default_segmentation_params['remove_small_holes']['area_threshold'],
                                   help="maximum threshold size of connected regions of 0")
    segmentation_args.add_argument('--local_max-min_distance',
                                   required=False, type=int,
                                   default=default_segmentation_params['peak_local_max']['min_distance'],
                                   help="minimum distance between two local maxima. Combats over-segmentation")
    segmentation_args.add_argument('--local_max-threshold_rel',
                                   required=False, type=float,
                                   default=default_segmentation_params['peak_local_max']['threshold_rel'],
                                   help="filter smaller maxima based on (this value * max(all_maxima)). "
                                   "Combats over-segmentation")
    segmentation_args.add_argument('--small_labels-min_size',
                                   required=False, type=int,
                                   default=default_segmentation_params['merge_small_labels']['min_size'],
                                   help="minimum size of an independent label; smaller labels will "
                                   "be merged to their largest neighbors. Combats over-segmentation")

    # TODO: Add args for classification
    # classification_args = classify_subparser.add_argument_group("Classification Algorithm Arguments")
    # classification_args.add_argument(...)

    # TODO: Add args for hardware
    # hardware_args = classify_subparser.add_argument_group("Hardware Arguments")
    # hardware_args.add_argument('--use_gpu', action='store_true', help='use GPU if tensorflow with CUDA installed')
    # hardware_args.add_argument('--gpu_device', required=False, default='0', type=str,
    #                            help='which GPU device to use, use an integer for CUDA, or mps for M1')

    config_subparser.add_argument('--param_dir', required=False, default='models', type=str,
                                  help="folder of model parameters")

    return parser

def main():
    args = get_arg_parser().parse_args()

    if args.command == 'version':
        print(version_str)
        return

    if args.verbose:
        from brachistools.io import logger_setup
        logger, _ = logger_setup()
    else:
        logger = logging.getLogger(__name__)

    # TODO: Assign devices
    # from brachistools import classification
    # device, gpu = classification.assign_device(use_tensorflow=True, gpu=args.use_gpu, device=...)

    if args.command == 'gui':
        if not GUI_ENABLED:
            print('GUI ERROR:', GUI_ERROR)
            if GUI_IMPORT:
                print('GUI FAILED: GUI dependencies may not be installed, to install, run')
                print('     pip install "brachistools[gui]"')
            sys.exit(-1)
        else:
            gui.run()

    if args.command == 'config':
        from configparser import ConfigParser

        try:
            config_path = 'config.ini'
            config = ConfigParser()
            config.read(config_path)
        except:
            logger.critical("Failed to open config file")
            sys.exit(-1)

        config.set('ModelParams', 'param_dir', args.param_dir)

        with open(config_path, 'w') as config_f:
            config.write(config_f)

    # Prepare images
    if args.dir and args.image_path:
        logger.critical("Cannot specify both --dir and --image_path")
        sys.exit(-1)

    if args.command == 'show':
        file_ext = 'XML'
    else:
        file_ext = ['PNG', 'JPG', 'JPEG']

    if args.dir:
        image_names = load_folder(args.dir,
            file_ext=file_ext, absolute_path=False,
            ignored_suffixes=args.ignored_suffixes)
        if len(image_names) > 1:
            image_names = tqdm(image_names)
    elif args.image_path:
        args.dir = str(Path(args.image_path).parent)
        image_names = [args.image_path]
    else:
        logger.critical("Input is not specified")
        sys.exit(-1)

    if not args.save_dir:
        args.save_dir = args.dir

    def savepath(fn):
        return os.path.join(args.save_dir, fn)

    if args.command == 'show':
        # from brachistools.io import HAVE_MATPLOTLIB
        # if not HAVE_MATPLOTLIB:
        #     logger.info("Showing segmentation XMLs requires matplotlib. Saving picture only")

        for seg_xml in image_names:
            try:
                root_fn, old_ext = os.path.splitext(seg_xml)
                if root_fn.endswith('_seg'):
                    root_fn = root_fn[:-4]

                tree = ET.parse(os.path.join(args.dir, seg_xml))
                labels, pic = xml_to_labels(tree, use_tqdm=len(image_names)==1)
                imsave(savepath(root_fn + '_xml2seg.PNG'), pic)
                if args.save_npy:
                    np.save(savepath(root_fn + '_xml2seg.npy'), labels)
            except Exception as e:
                logger.critical(
                    "Failed to transform segmentation XML '%%' "
                    "due to exception: %%", seg_xml, e)

    if args.command == 'segment':
        segment_params = default_segmentation_params.copy()
        segment_params['vahadane:sparsity_regularizer'] = args.vahadane_sparsity_regularizer
        segment_params['equalize_adapthist:clip_limit'] = args.equalize_adapthist_clip_limit
        segment_params['remove_small_objects:min_size'] = args.small_objects_min_size
        segment_params['remove_small_holes:area_threshold'] = args.small_holes_area_threshold
        segment_params['peak_local_max:min_distance'] = args.local_max_min_distance
        segment_params['peak_local_max:threshold_rel'] = args.local_max_threshold_rel
        segment_params['merge_small_labels:min_size'] = args.small_labels_min_size

        for fn in image_names:
            try:
                image = imread(os.path.join(args.dir, fn))
                nucleus, labeled_nucleus = segmentation_pipeline(image, segment_params)

                root, old_ext = os.path.splitext(fn)
                imsave(savepath(f"{root}_mask.{args.save_format}"), img_as_ubyte(nucleus))
                labels_to_xml(labeled_nucleus).write(savepath(f"{root}_seg.xml"))

                if args.save_npy:
                    np.save(savepath(root + '_mask.npy'), nucleus)
                    np.save(savepath(root + '_mask_labels.npy'), labeled_nucleus)
            except Exception as e:
                logger.critical(
                    "Failed to segmentation picture '%%' "
                    "due to exception: %%", fn, e)

    if args.command == 'classify':
        with open('./result.csv', 'w', newline='') as csvfile:
            fieldnames = ['Image Name', 'Predict', 'Confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for fn in image_names:
                image = imread(fn)
                predict_class, confidence_score = classification_pipeline(image)
                writer.writerow(
                    {'Image Name': fn, 'Predict': predict_class, 'Confidence': confidence_score})

if __name__ == "__main__":
    main()
