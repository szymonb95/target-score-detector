import HomographicMatcher as matcher
import VisualAnalyzer as visuals
import GroupingMetre as grouper
import HitsManager as hitsMngr
import Geometry2D as geo2D
import numpy as np
import cv2

class VideoAnalyzer:
    def __init__(self, videoPath, model, bullseye, ringsAmount, diamPx):
        '''
        {String} videoName - The path of the video to analyze
        {Numpy.array} model - An image of the target that appears in the video
        {Tuple} bullseye - (
                              {Number} x coordinate of the bull'seye location in the model image,
                              {Number} y coordinate of the bull'seye location in the model image
                           )
        {Number} ringsAmount - Amount of rings in the target
        {Number} diamPx - The diameter of the most inner ring in the target image [px]
        '''

        self.cap = cv2.VideoCapture(videoPath)
        # camera capture
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        _, test_sample = self.cap.read()
        frameSize = test_sample.shape
        self.rings_amount = ringsAmount
        self.inner_diam = diamPx
        self.model = model
        self.frame_h, self.frame_w, _ = frameSize
        self.matcher = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.004, sigma=1.2)

        # calculate anchor points and model features
        self.anchor_points, self.pad_model = geo2D.zero_pad_as(self.model, frameSize)
        self.bullseye_point = bullseye
        anchor_a = self.anchor_points[0]
        bullseye_anchor = (anchor_a[0] + bullseye[0],anchor_a[1] + bullseye[1])
        self.anchor_points.append(bullseye_anchor)
        self.anchor_points = np.float32(self.anchor_points).reshape(-1, 1, 2)

        self.model_gray = cv2.cvtColor(model, cv2.COLOR_RGB2GRAY)
        self.model_gray = cv2.GaussianBlur(self.model_gray, (5, 5), 0)
        self.model_keys, self.model_desc = self.matcher.detectAndCompute(self.model_gray, None)
        eypointimage = cv2.drawKeypoints(self.model_gray, self.model_keys, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('model_keypoints', eypointimage)

    def _analyze_frame(self, frame):
        '''
        Analyze a single frame.

        Parameters:
            {Numpy.array} frame - The frame to analyze

        Returns:
            {Tuple} (
                        {Number} x coordinate of the bull'seye point in the target,
                        {Number} y coordinate of the bull'seye point in the target,
                    ),
            {list} [
                       {tuple} (
                                   {tuple} (
                                              {Number} x coordinates of the hit,
                                              {Number} y coordinates of the hit
                                           ),
                                   {Number} The hit's score according to the target's data
                               )
                       ...
                   ],
        '''

        # set default analysis meta-data
        scoreboard = []
        scores = []
        warped_frame = None
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_shape = frame_gray.shape
        # frame_gray = cv2.resize(frame_gray, (int(frame_shape[1] * 0.8), int(frame_shape[0] * 0.8)) )

        # find a match between the model image and the frame
        matches, (frame_keys, frame_desc) = matcher.ratio_match(self.matcher, self.model_desc, frame_gray, .8)

        # start calculating homography
        if len(matches) >= 4:
            homography, mask = matcher.calc_homography(self.model_keys, frame_keys, matches)

            # check if homography succeeded and start warping the model over the detected object
            if type(homography) != type(None):
                matchesMask = mask.ravel().tolist()
                h,w = self.model_gray.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                warped_transform = cv2.perspectiveTransform(pts, homography)
                img2 = cv2.polylines(np.copy(frame_gray), [np.int32(warped_transform)], True, 255, 3, cv2.LINE_AA)
                warped_transform = np.append(warped_transform, np.array(self.bullseye_point).reshape(1,1,2), axis=0)
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2
                )
                img3 = cv2.drawMatches(self.model_gray, self.model_keys, img2, frame_keys, matches, None, **draw_params)
                cv2.imshow('matches', img3)

                warped_vertices, warped_edges = geo2D.calc_vertices_and_edges(warped_transform)

                # check if homography is good enough to continue
                # if matcher.is_true_homography(warped_vertices, warped_edges, (self.frame_w, self.frame_h), .2):
                # warp the input image over the filmed object and calculate the scale difference
                warped_frame_gray = cv2.warpPerspective(frame_gray, homography, (w, h), flags=cv2.WARP_INVERSE_MAP)
                warped_frame = cv2.warpPerspective(frame, homography, (w, h), flags=cv2.WARP_INVERSE_MAP)
                # cv2.imshow('warped_img', warped_frame)
                scale = geo2D.calc_model_scale(warped_edges, self.model.shape)
                
                # process image
                sub_target = visuals.subtract_background(self.model_gray, cv2.resize(warped_frame_gray, (self.model.shape[1], self.model.shape[0])))
                pixel_distances = geo2D.calc_distances_from(self.model.shape, self.bullseye_point)
                estimated_warped_radius = self.rings_amount * self.inner_diam * scale[2]
                cv2.imshow('sub_target', sub_target)
                
                proj_contours = visuals.detect_hit_contours(sub_target)
                
                suspect_hits = visuals.find_suspect_hits(proj_contours, warped_vertices, scale)

                # calculate hits and draw circles around them
                scoreboard = hitsMngr.create_scoreboard(suspect_hits, scale, self.rings_amount, self.inner_diam)

                # insert warped frame in original
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*0, (0x0, 0x0, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*1, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*2, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*3, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*4, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*5, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*6, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*7, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*8, (0xff, 0xff, 0xff), 4)
                # cv2.circle(self.model, (int(self.bullseye_point[0]), int(self.bullseye_point[1])), 21+32*9, (0xff, 0xff, 0xff), 4)
                frame[:w,:h] = warped_frame

        return self.bullseye_point, scoreboard

    def analyze(self, outputName, sketcher):
        '''
        Analyze a video completely and output the same video, with additional data written in it.

        Parameters:
            {String} outputName - The path of the output file
            {Sketcher} sketcher - A Sketcher object to use when writing the data to the output video
        '''

        # set output configurations
        frame_size = (self.frame_w, self.frame_h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outputName, fourcc, 24.0, frame_size)

        while True:
            ret, frame = self.cap.read()

            if ret:
                bullseye, scoreboard = self._analyze_frame(frame)
                
                # increase reputation of consistent hits
                # or add them as new candidates
                for hit in scoreboard:
                    hitsMngr.sort_hit(hit, 30, 15)
                
                # decrease reputation of inconsistent hits
                hitsMngr.discharge_hits()
                
                # stabilize all hits according to the slightly shifted bull'seye point
                if type(bullseye) != type(None):
                    hitsMngr.shift_hits(bullseye)

                # reference hit groups
                candidate_hits = hitsMngr.get_hits(hitsMngr.CANDIDATE)
                verified_hits = hitsMngr.get_hits(hitsMngr.VERIFIED)

                # extract grouping data
                grouping_contour = grouper.create_group_polygon(frame, verified_hits)
                has_group = type(grouping_contour) != type(None)
                grouping_diameter = grouper.measure_grouping_diameter(grouping_contour) if has_group else 0

                # write meta data on frame
                sketcher.draw_data_block(frame)
                verified_scores = [h.score for h in verified_hits]
                arrows_amount = len(verified_scores)
                sketcher.type_arrows_amount(frame, arrows_amount, (0x0,0x0,0xff))
                sketcher.type_total_score(frame, sum(verified_scores), arrows_amount * 10, (0x0,189,62))
                sketcher.type_grouping_diameter(frame, grouping_diameter, (0xff,133,14))
                
                # mark hits and grouping
                sketcher.draw_grouping(frame, grouping_contour)
                sketcher.mark_hits(frame, candidate_hits, foreground=(0x0,0x0,0xff),
                                   diam=2, withOutline=False, withScore=False)
                
                sketcher.mark_hits(frame, verified_hits, foreground=(0x0,0xff,0x0),
                                   diam=5, withOutline=True, withScore=True)
                
                # display
                frame_resized = cv2.resize(frame, (1153, 648))
                cv2.imshow('Analysis', frame_resized)
                
                # write frame to output file
                out.write(frame)
                
                if cv2.waitKey(1) & 0xff == 27:
                    break
            else:
                print('Video stream is over.')
                break
                
        # close window properly
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)