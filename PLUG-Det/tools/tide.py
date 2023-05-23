from tidecv import TIDE, datasets

tide = TIDE()
tide.evaluate(datasets.COCO(path='/media/h/M/dataset/AITOD/annotations/aitod_test_v1.json'), datasets.COCOResult('/media/h/M/dataset/AITOD/work_dirs/centernet/results_val1.bbox.json'), mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot()       # Show a summary figure. Specify a folder and it'll output a png to that folder.
