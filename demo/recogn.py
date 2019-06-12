from tkinter.messagebox import showinfo
import tkinter
import PIL.Image, PIL.ImageTk
import time

import torch

from utils.helper import calculate_cosine_similarity
time_start = 0
class App:
    def __init__(self, window, window_title, video_source, database, mrcnn_api,gaitnet_api):
        self.database = database
        self.mrcnn_api, self.gaitnet_api = mrcnn_api,gaitnet_api

        self.is_record = False

        self.frames = []
        self.segs = []

        self.window = window
        self.window.title(window_title)

        self.vid = video_source

        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.btn_capture=tkinter.Button(window, text="Capture", width=50, command=self.record)
        self.btn_recogn = tkinter.Button(window, text="Recognize", width=50, command=self.recogn)

        self.canvas.pack()
        self.btn_capture.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_recogn.pack(anchor=tkinter.CENTER, expand=True)

        self.delay = 1
        self.update()

        self.index = 0

        self.window.mainloop()



    def recogn(self):
        if self.database == {}:
            showinfo("Error", "Database is empty!")
            return

        try:
            with torch.no_grad():
                self.segs = self.mrcnn_api.get_seg_batch(self.frames, 10)
                feature = self.gaitnet_api.main(self.segs)  # from video to feature
                torch.cuda.empty_cache()
        except:
            print("Video processing failed. Please record again.")
            self.frames.clear()
            self.is_record = True
            return

        name = ''
        value = 0
        for k,v in self.database.items():
            current_score = calculate_cosine_similarity(v,feature)
            print(k, round(current_score * 100, 3))
            if current_score > value:
               name = k
               value = current_score
        showinfo("Result", "{}:{}%".format(name,round(value*100,3)))
        self.frames.clear()
    def record(self):
        # Get a frame from the video source
        global time_start
        if self.btn_capture['text'] == "Capture":
            self.btn_capture['text'] = 'Stop'
            # events
            self.frames.clear()
            self.is_record = True
        else:
            self.btn_capture['text'] = 'Capture'
            self.is_record = False

    def update(self):
        if self.is_record is False and len(self.frames) != 0:  # replay
            if self.index >= len(self.frames):
                self.index = 0
            frame = self.frames[self.index]
            self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=tkinter.NW)
            self.index += 1
        else:  # live
            exist, frame = self.vid.get_frame()
            if exist:  # if get the frame
                self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.frame, anchor=tkinter.NW)
                if self.is_record: # if error raised by above line
                    self.frames.append(frame)

        self.window.after(self.delay, self.update)



def main(database,video_source, mrcnn_api,gaitnet_api, args__):
    App(tkinter.Toplevel(), "GaitNet Recognition", video_source, database,mrcnn_api,gaitnet_api)