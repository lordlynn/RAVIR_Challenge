%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Final Project - Average Neighbors
% Filename: avgNeighbors.m
% Author: Zac Lynn
% Date: 4/2/2023
% Instructor: Dr. Rhodes
% Description: This file contains code that can apply neighborhood averages
%                   given an image, a mask to apply the averaging, and
%                   values for i and j on which to center the mask. For a
%                   basic average the mask should be all ones. For
%                   laplacians, use the mask accordingly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [avg, total]=avgNeighbors(img, mask, i, j)
% Uses the supplied mask to average the neighbors of the pixel found at
% (i, j)

    sum = 0.0;
    count = 0;
    
    yUnder = 0; % could use to apply values to the pixels that are out of bounds
    yOver = 0;
    xUnder = 0;
    xOver = 0;
    
    yMin = j - floor(size(mask, 1) / 2);
    if (yMin < 1) 
       yUnder = abs(yMin) + 1;
       yMin = 1;
    end
    
    yMax = j + floor(size(mask, 1) / 2);
    if  (yMax > size(img, 1))
        yOver = yMax - size(img, 1);
        yMax = size(img, 1);
    end
    
    xMin = i - floor(size(mask, 2) / 2);
    if (xMin < 1) 
       xUnder = xMin - 1;
       xMin = 1;
    end
    
    xMax = i + floor(size(mask, 2) / 2);
    if  (xMax > size(img, 2))
        xOver = xMax - size(img, 2);
        xMax = size(img, 2);
    end
    
    for x = xMin:xMax
        for y = yMin:yMax
            sum = sum + img(y, x) * mask(yMax - y + 1, xMax - x + 1);
            count = count + mask(yMax - y + 1, xMax - x + 1);
        end
    end
    
    avg = sum / count;
    total = sum;
end


