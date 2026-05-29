import serial
import math
import numpy as np
import time
import tkinter as tk
import warnings
import threading
    

class YDLidarX2:
    
    def __init__(self, port, chunk_size=2000):
        self.__version = 1.03
        self._port = port                # string denoting the serial interface
        self._ser = None
        self._chunk_size = chunk_size    # reasonable range: 1000 ... 10000
        self._min_range = 10			 # minimal measurable distance
        self._max_range = 8000			 # maximal measurable distance
        self._max_data = 20              # maximum number of datapoints per angle
        self._out_of_range = 32768       # indicates invalid data
        self._is_connected = False
        self._is_scanning = False
        self._scan_is_active = False
        self._availability_flag = False
        self._debug_level = 0
        self._error_cnt = 0
        self._lock = threading.Lock()
        self._last_chunk = None
        # 2D array capturing the distances for angles from 0 to 359
        self._distances = np.array([[self._out_of_range for _ in range(self._max_data)] for l in range(360)], dtype=np.uint32)
        # 1D array capturing the number of measurements for angles from 0 to 359
        self._distances_pnt = np.array([0 for _ in range(360)], dtype=np.uint32)
        # predefined list of angle corrections for distances from 0 to 8000
        self._corrections = np.array([0.0] + [math.atan(21.8*((155.3-dist)/(155.3*dist)))*(180/math.pi) for dist in range(1, 8001)])
        # measured distances for angles from 0 to 359
        self._result = np.array([self._out_of_range for _ in range(360)], dtype=np.int32)
        # operating variables for plot functions
        self._org_x, self._org_y = 0, 0
        self._scale_factor = 0.2
        # constants -----
        # array of boundary angles for each sector
        self._sector40_lst = np.array([angle for angle in range(0, 361, 9)], dtype=np.int32)
        self._sector20_lst = np.array([angle for angle in range(0, 361, 18)], dtype=np.int32)
        # array of midpoint angles for each sector
        self._sector40_midpoints = np.arange(4.5, 360.0,  9.0)
        self._sector20_midpoints = np.arange(9.0, 360.0, 18.0)
        # arrays with pre-calculated sinus and cosinus
        self._sin_x = np.array([math.sin(x * math.pi / 180) for x in range(-180, 180)])
        self._cos_x = np.array([math.cos(x * math.pi / 180) for x in range(-180, 180)])
        
        
    def connect(self):
        """ Connects on serial interface """
        if not self._is_connected:
            try:
                self._ser = serial.Serial(self._port, 115200, timeout = 1)
                self._is_connected = True
            except Exception as e:
                print(e)
                self._is_connected = False
        else:
            warnings.warn("connect: LiDAR already connected", RuntimeWarning)
        return self._is_connected
    
    
    def disconnect(self):
        """ Disconnects the serial interface """
        if self._is_connected:
            self._ser.close()
            self._is_connected = False
        else:
            warnings.warn("disconnect: LiDAR not connected", RuntimeWarning)
            
            
    def start_scan(self):
        """ Starts a thread to run the scan process. """
        if not self._is_connected:
            warnings.warn("start_scan: LiDAR not connected", RuntimeWarning)
            return False
        self._is_scanning = True
        self._scan_thread = threading.Thread(target = self._scan)
        self._scan_thread.start()
        self._availability_flag = False
        return True
    
    
    def stop_scan(self):
        """ Stops the thread running the scan process. """
        if not self._is_scanning:
            warnings.warn("stop_scan: LiDAR is not scanning", RuntimeWarning)
            return False
        else:
            self._is_scanning = False
            while not self._scan_is_active:
                time.sleep(0.1)
            time.sleep(self._chunk_size / 6000)		# wait for the last chunk to finish reading
        return True
    
    
    def _scan(self):
        """ Core routine to retrieve and decode lidar data.
            Availaility flag is set after each successful decoding process. """
        self._scan_is_active = True
        while self._is_scanning:
            # Retrieve data
            data = self._ser.read(self._chunk_size).split(b"\xaa\x55")
            if self._last_chunk is not None:
                data[0] = self._last_chunk + data[0]
            self._last_chunk = data.pop()
            # Clear array for new scan
            distances_pnt = np.array([0 for _ in range(360)], dtype=np.uint32)
            error_cnt = 0
            # Decode data
            for idx, d in enumerate(data):
                # Reasonable length of the data slice?
                l = len(d)
                if l < 10:
                    error_cnt += 1
                    if self._debug_level > 0:
                        print("Idx:", idx, "ignored - len:", len(d))
                    continue
                # Get sample count and start and end angle
                sample_cnt = d[1]
                # Do we have any samples?
                if sample_cnt == 0:
                    error_cnt += 1
                    if self._debug_level > 0:
                        print("Idx:", idx, "ignored - sample_cnt: 0")
                    continue
                # Get start and end angle
                start_angle = ((d[2] + 256 * d[3]) >> 1) / 64
                end_angle = ((d[4] + 256 * d[5]) >> 1) / 64
 
                # Start data block
                if sample_cnt == 1:
                    dist = round((d[8] + 256*d[9]) / 4)
                    if self._debug_level > 1:
                        print("Start package: angle:", start_angle, "   dist:", dist)
                    if dist > self._min_range:
                        if dist > self._max_range: dist = self._max_range
                        angle = round(start_angle + self._corrections[dist])
                        if angle < 0: angle += 360
                        if angle >= 360: angle -= 360
                        self._distances[angle][distances_pnt[angle]] = dist
                        if distances_pnt[angle] < self._max_data - 1:
                            distances_pnt[angle] += 1
                        else:
                            if self._debug_level > 0:
                                print("Idx:", idx, " - pointer overflow")
                            error_cnt += 1

                # Cloud data block
                else:
                    if start_angle == end_angle:
                        if self._debug_level > 0:
                            print("Idx:", idx, "ignored - start angle equals end angle for cloud package")
                        error_cnt += 1
                        continue
                    if l != 8 + 2 * sample_cnt:
                        if self._debug_level > 0:
                            print("Idx:", idx, "ignored - len does not match sample count - len:", l, " - sample_cnt:", sample_cnt)
                        error_cnt += 1
                        continue
                    if self._debug_level > 1:
                        print("Cloud package: angle:", start_angle, "-", end_angle)
                    if end_angle < start_angle:
                        step_angle = (end_angle + 360 - start_angle) / (sample_cnt - 1)
                    else:
                        step_angle = (end_angle - start_angle) / (sample_cnt - 1)
                    pnt = 8
                    while pnt < l:
                        dist = round((d[pnt] + 256*d[pnt+1]) / 4)
                        if dist > self._min_range:
                            if dist > self._max_range: dist = self._max_range
                            angle = round(start_angle + self._corrections[dist])
                            if angle < 0: angle += 360
                            if angle >= 360: angle -= 360
                            self._distances[angle][distances_pnt[angle]] = dist
                            if distances_pnt[angle] < self._max_data - 1:
                                distances_pnt[angle] += 1
                            else:
                                if self._debug_level > 0:
                                    print("Idx:", idx, " - pointer overflow")
                                error_cnt += 1
                        start_angle += step_angle
                        if start_angle >= 360: start_angle -= 360
                        pnt += 2
            # calculate result
            if self._debug_level > 0 and error_cnt > 0:
                print("Error cnt:", error_cnt)
            
            for angle in range(360):
                if distances_pnt[angle] == 0:
                    self._result[angle] = self._out_of_range
                else:
                    self._result[angle] = self._distances[angle][:distances_pnt[angle]].mean()
            self._error_cnt = error_cnt
            self._availability_flag = True
        # end of decoding loop        
        self._scan_is_active = False
        
        
    def get_data(self):
        """ Returns an array of distance data (360 values, one for each degree).
            Resets availability flag"""
        if not self._is_scanning:
            warnings.warn("get_data: Lidar is not scanning", RuntimeWarning)
        self._lock.acquire()
        distances = self._result.copy()
        self._availability_flag = False
        self._lock.release()
        return distances
    
    def _get_sector40_lst(self):
        """ Returns an array of the border angles for each of the 40 sectors. """
        return self._sector40_lst
    
    def _get_sector20_lst(self):
        """ Returns an array of the border angles for each of the 20 sectors. """
        return self._sector20_lst
    
    def _get_sector40_midpoints(self):
        """ Returns midpoint angles of all 40 sectors """
        return self._sector40_midpoints
    
    def _get_sector20_midpoints(self):
        """ Returns midpoint angles of all 20 sectors """
        return self._sector20_midpoints

    def _available(self):
        """ Indicates whether a new dataset is available """
        return self._availability_flag
    
    def _get_error_cnt(self):
        """ Returns the error count of last data chunk """
        return self._error_cnt
    
    def _get_connected(self):
        """ Returns the status of the connection to the serial interface """
        return self._is_connected

    def _get_scanning(self):
        """ Returns indicator showing whether scanning is active """
        return self._is_scanning
    
    def _get_version(self):
        """ Returns version number of the driver software """
        return self.__version
    
    def _get_out_of_range(self):
        """ Returns out-of-range value, indicating an invalid data item """
        return self._out_of_range
    
    def _get_scale_factor(self):
        return self._scale_factor
    
    def _set_scale_factor(self, f):
        if 0 < f <= 1:
            self._scale_factor = f
        else:
            warnings.warn("set_scale_factor: statement ignored!\nValue must be in range 0 ... 1", RuntimeWarning)
    

    is_connected = property(_get_connected)
    is_scanning = property(_get_scanning)
    out_of_range = property(_get_out_of_range)
    available = property(_available)
    error_cnt = property(_get_error_cnt)
    sector40_lst = property(_get_sector40_lst)
    sector20_lst = property(_get_sector40_lst)
    sector40_midpoints = property(_get_sector40_midpoints)
    sector20_midpoints = property(_get_sector20_midpoints)
    scale_factor = property(_get_scale_factor, _set_scale_factor)
    __version__ = property(_get_version)