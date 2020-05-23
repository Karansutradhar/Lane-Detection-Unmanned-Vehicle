import sys
import numpy as np
import cv2
import math
import argparse


class VideoReader:
    def __init__(self, input_video):
        self.fileName = input_video
        self.reader = cv2.VideoCapture(self.fileName)
        if not self.reader.isOpened():
            sys.stdout.write("\nUnable to locate the file %s."
                             "\nPlease ensure that the filename is spelled correctly and that "
                             "the file is in the correct location.\n" % self.fileName)
            exit(0)
        self.window_name = "Frame"
        self.frames = []
        self.display_frames = []
        self.scale = 0.5
        self.width = int((self.reader.get(3) * self.scale) // 2) * 2
        self.height = int((self.reader.get(4) * self.scale) // 2) * 2
        self.video_output_name = "LaneOverlay.mp4"
        self.writer = cv2.VideoWriter(self.video_output_name, cv2.VideoWriter_fourcc('H', '2', '6', '4'), 15,
                                      (self.width, self.height))

        self.read_video()
        self.process_frames()
        self.play(self.display_frames)

    def read_video(self):
        ret, frame = self.reader.read()
        while ret and len(self.frames) < 9999:
            self.frames.append(cv2.resize(frame, dsize=None, fx=self.scale, fy=self.scale))
            ret, frame = self.reader.read()

    def process_frames(self):
        horizon = []
        turn_array = []
        horizon_count = 16
        turn_count = 10
        line_list = []  # x1b, y1b, x1t, y1t, x2b, y2b, x2t, y2t
        for i, frame in enumerate(self.frames):
            # Find yellow regions corresponding to the sides of the lane
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([18, 64, 128])
            upper_yellow = np.array([30, 192, 255])
            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            yellow_mask = cv2.dilate(yellow_mask, dilate_kernel)
            yellow_mask = cv2.erode(yellow_mask, erode_kernel)
            frame_yellow_enhanced = cv2.bitwise_or(frame, cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2RGB))

            # Enhance white areas with histogram equalization
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            equalized_image = np.copy(blur_image)
            equalized_image[:int(self.height * 0.6)] = 0
            equalized_image = cv2.equalizeHist(equalized_image)
            ret, white_mask_eq = cv2.threshold(equalized_image, 253, 255, cv2.THRESH_BINARY)
            white_mask_eq[:int(self.height * 0.6)] = 0
            frame_enhanced = cv2.bitwise_or(frame_yellow_enhanced, cv2.cvtColor(white_mask_eq, cv2.COLOR_GRAY2RGB))

            # Grayscale and blur
            gray_image = cv2.cvtColor(frame_enhanced, cv2.COLOR_RGB2GRAY)
            blur_image = cv2.bilateralFilter(gray_image, 7, 45, 75)
            blur_image = cv2.GaussianBlur(blur_image, (5, 5), 0)

            # Find white regions corresponding to the sides of the lane
            ret, white_mask = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            white_mask = cv2.erode(white_mask, erode_kernel)
            white_mask = cv2.dilate(white_mask, dilate_kernel)

            # Find gray regions corresponding to the surface of the road
            hsv_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_gray = np.array([0, 0, 32])
            upper_gray = np.array([180, 40, 192])
            gray_mask_hsv = cv2.inRange(hsv_gray, lower_gray, upper_gray)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gray_mask_hsv = cv2.erode(gray_mask_hsv, erode_kernel)
            gray_mask_hsv = cv2.dilate(gray_mask_hsv, dilate_kernel)

            # Exclude the middle of the road
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            horizon_image = np.copy(gray_mask_hsv)
            middle_road = 255 - cv2.erode(gray_mask_hsv, erode_kernel)
            gray_mask_hsv = cv2.bitwise_and(gray_mask_hsv, middle_road)

            # Canny edge detection and crop image
            canny_image = cv2.Canny(blur_image, 50, 150, 3)
            canny_image[:int(self.height * 0.5)] = 0

            # Mask the lanes for just the road and lane divisions
            combined_mask = cv2.bitwise_or(gray_mask_hsv, white_mask)
            masked_image = cv2.bitwise_and(canny_image, combined_mask)

            # Search for Hough lines
            # hough_lines = cv2.HoughLinesP(masked_image, 4, np.pi * 2 / 20, 30, minLineLength=12, maxLineGap=40)
            hough_lines_1 = cv2.HoughLines(masked_image, 2, np.pi*2/90, 60, min_theta=np.pi*0.25, max_theta=np.pi*0.35)
            hough_lines_2 = cv2.HoughLines(masked_image, 2, np.pi*2/90, 60, min_theta=np.pi*0.70, max_theta=np.pi*0.90)

            # Average Hough lines
            fitted_lines = []
            valid_horizons = [h for h in horizon if h != -1]
            for line_set in [hough_lines_1, hough_lines_2]:
                if line_set is not None:
                    line = []
                    for j in range(len(line_set)):
                        rho = line_set[j][0][0]
                        theta = line_set[j][0][1]
                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
                        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
                        # Draw all detected Hough lines with:  cv2.line(line_image, pt1, pt2, (255, 0, 0), 5)
                        parameters = np.polyfit((pt1[0], pt2[0]), (pt1[1], pt2[1]), 1)
                        slope = parameters[0]
                        intercept = parameters[1]
                        line.append((slope, intercept))
                    fitted_lines.append(line)
            fit_average = [np.average(ln, axis=0) for ln in fitted_lines]
            lane_points = []
            for j in range(len(fit_average)):
                slope, intercept = fit_average[j]
                y1 = self.height
                average_horizon = int(np.mean(valid_horizons)) if len(valid_horizons) > 0 else 0
                y2 = int(average_horizon)
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
                lane_points.append((x1, y1, x2, y2))

            # Find the horizon
            if len(lane_points) == 2:
                row = int(self.height * 0.55)
                while row < int(self.height * 0.75) and np.sum(horizon_image[row]) / 255 / self.width < 0.26:
                    row += 1
                # line list structure:  x1b, y1b, x1t, y1t, x2b, y2b, x2t, y2t
                x1b = lane_points[0][0]
                y1b = lane_points[0][1]
                x1t = lane_points[0][2]
                y1t = lane_points[0][3]
                x2b = lane_points[1][0]
                y2b = lane_points[1][1]
                x2t = lane_points[1][2]
                y2t = lane_points[1][3]
                m1, b1 = fit_average[0]
                m2, b2 = fit_average[1]
                while x2t < x1t + int(0.05 * (x2b - x1b)) and y1t < self.height:
                    y1t += 2
                    y2t = y1t
                    x1t = int((y1t - b1) / m1)
                    x2t = int((y2t - b2) / m2)
                next_two_lines = [x1b, y1b, x1t, y1t, x2b, y2b, x2t, y2t]
            else:
                row = -1
                next_two_lines = -1
            if len(horizon) < horizon_count:
                horizon.append(row)
                line_list.append(next_two_lines)
            else:
                for j in range(len(horizon) - 1):
                    horizon[j] = horizon[j + 1]
                    line_list[j] = line_list[j + 1]
                horizon[horizon_count - 1] = row
                line_list[horizon_count - 1] = next_two_lines

            # Generate final image of road with lane overlay
            valid_lines = [ln for ln in line_list if ln != -1]
            if len(valid_lines) > 0:
                x1b, y1b, x1t, y1t, x2b, y2b, x2t, y2t = np.mean(np.array(valid_lines), axis=0).astype(np.int)

                # Draw lanes onto the road
                final_image = self.draw_lines(frame, (x1b, y1b, x1t, y1t, x2b, y2b, x2t, y2t), 0.6)

                # ** Turn Prediction ** #
                # Unwarp lanes to predict turn direction
                width_increase = 10
                new_lines = [[x1b - width_increase, y1b, x1t - width_increase // 2, y1t,
                              x1b + width_increase, y1b, x1t + width_increase // 2, y1t],
                             [x2b - width_increase, y2b, x2t - width_increase // 2, y2t,
                              x2b + width_increase, y2b, x2t + width_increase // 2, y2t]]
                turns = ""
                for nx1b, ny1b, nx1t, ny1t, nx2b, ny2b, nx2t, ny2t in new_lines:
                    source_points = np.array([[[nx1b, ny1b], [nx1t, ny1t], [nx2t, ny2t], [nx2b, ny2b]]])
                    dest_points = np.array([[[self.width * 3 // 7, self.height],
                                             [self.width * 3 // 7, 0],
                                             [self.width * 4 // 7, 0],
                                             [self.width * 4 // 7, self.height]]])
                    h_mat = cv2.findHomography(source_points, dest_points)[0]
                    total_mask = cv2.bitwise_and(cv2.bitwise_or(yellow_mask, white_mask_eq), middle_road)
                    total_mask[:int(self.height * 0.55)] = 0
                    unwarped_image = cv2.warpPerspective(total_mask, h_mat, (self.width, self.height),
                                                         flags=cv2.INTER_NEAREST)
                    unwarped_image[:, :int(self.width * 0.4)] = 0
                    unwarped_image[:, int(self.width * 0.6):] = 0
                    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    unwarped_image = cv2.erode(unwarped_image, erode_kernel)

                    # Fit a polynomial to the side of each lane
                    fit_x = []
                    fit_y = []
                    for yp in range(self.height):
                        for xp in range(self.width):
                            if unwarped_image[yp, xp]:
                                fit_x.append(xp)
                                fit_y.append(yp)
                    # Use the second derivative of the polynomial to determine if (and how) the lane is turning
                    if len(fit_x) > 0:
                        try:
                            c = [p for p in np.polyfit(fit_y, fit_x, 2)]
                        except np.linalg.LinAlgError:
                            c = [0, 0, -1]
                        c.reverse()
                        if -0.003 <= c[2] < -0.0008:
                            turns = "%s%s" % (turns, "L")
                        elif 0.0008 <= c[2] < 0.003:
                            turns = "%s%s" % (turns, "R")
                            steps = 4
                            for yp in range(0, self.height-steps, steps):
                                x = int(c[2] * (yp ** 2) + c[1] * yp + c[0])
                                next_x = int(c[2] * ((yp + steps) ** 2) + c[1] * (yp + steps) + c[0])
                                cv2.line(unwarped_image, (x, yp), (next_x, yp + steps), 128, thickness=5)
                        elif abs(c[2]) <= 0.001:
                            turns = "%s%s" % (turns, "_")

                # Update the smoothing filter for turn prediction
                if len(turns) == 2:
                    if turns == "LL" or turns == "RR":
                        t = 1 if "R" in turns else -1
                    else:
                        t = 0
                else:
                    t = 0
                if len(turn_array) < turn_count:
                    turn_array.append(t)
                else:
                    for j in range(len(turn_array) - 1):
                        turn_array[j] = turn_array[j + 1]
                    turn_array[turn_count-1] = t

                # Use the turn prediction array to display the turn direction of the lane (if it is turning)
                if len(turn_array) > int(turn_count * 3 / 4):
                    average_turn = np.mean(turn_array)
                    turn_alpha = min(max((abs(average_turn) - 0.15) * 6, 0), 1)
                    final_image = self.draw_arrow(final_image, sign=1 if average_turn > 0 else -1,
                                                  alpha=0.75 * turn_alpha)
            else:
                final_image = np.copy(frame)

            # Display frames
            result = np.copy(final_image)
            if len(result.shape) < 3 or result.shape[2] == 1:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            display_top = frame
            display_bottom = result
            image_display = np.concatenate((display_top, display_bottom), axis=1)
            self.writer.write(final_image)
            self.display_frames.append(image_display)
        self.writer.release()

    @staticmethod
    def draw_lines(image, params, alpha):
        lane_image = np.copy(image)
        overlay_image = np.zeros_like(image)
        x1b, y1b, x1t, y1t, x2b, y2b, x2t, y2t = params
        cv2.fillPoly(overlay_image, np.array([[[x1b, y1b], [x1t, y1t], [x2t, y2t], [x2b, y2b]]]), (64, 64, 255))
        cv2.line(overlay_image, (x1b, y1b), (x1t, y1t), (64, 192, 255), 3)
        cv2.line(overlay_image, (x2b, y2b), (x2t, y2t), (64, 192, 255), 3)
        lane_image[np.where(overlay_image)] = overlay_image[np.where(overlay_image)]
        final_image = np.copy(image)
        cv2.addWeighted(lane_image, alpha, final_image, 1 - alpha, 0, final_image)
        return final_image

    def draw_arrow(self, image, sign=1, alpha=0.6):
        arrow_image = np.copy(image)
        overlay_image = np.zeros_like(image)
        x_l = self.width // 2 - sign * 40
        x_m = self.width // 2 + sign * 10
        x_r = self.width // 2 + sign * 40
        y_tt = int(self.height * 0.4) - 80
        y_mt = y_tt + 20
        y_mm = y_mt + 10
        y_mb = y_mm + 10
        y_bb = y_mb + 20
        polygon = np.array([[[x_l, y_mt], [x_m, y_mt], [x_m, y_tt], [x_r, y_mm], [x_m, y_bb], [x_m, y_mb],
                             [x_l, y_mb], [x_l, y_mt]]])
        cv2.fillPoly(overlay_image, polygon, (0, 192, 0))
        cv2.polylines(overlay_image, polygon, True, (10, 10, 255), thickness=2)
        arrow_image[np.where(overlay_image)] = overlay_image[np.where(overlay_image)]
        final_image = np.copy(image)
        cv2.addWeighted(arrow_image, alpha, final_image, 1 - alpha, 0, final_image)
        return final_image

    def play(self, frames):
        try:
            while True:
                for frame in frames:
                    cv2.imshow(self.window_name, frame)
                    cv2.waitKey(50)
        except KeyboardInterrupt:
            cv2.destroyWindow(self.window_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ENPM673 - Robot Perception - Project 2:  Lane Detection")
    parser.add_argument('video', type=str, help="Video from which to perform lane detection")
    v = argparse.Namespace()
    args = parser.parse_args(namespace=v)
    VideoReader(v.video)
