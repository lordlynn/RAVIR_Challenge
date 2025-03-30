%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: Final Project - Data Augmentation
% Filename: DataAugmentation.m
% Author: Zac Lynn
% Date: 4/2/2023
% Instructor: Dr. Rhodes
% Description: This file performs normalization and data augmentation for
%                   the supplied input images. Data is augmented using the
%                   7 non-destructive transforms, global brightness
%                   changes and 5x5 neighborhood blurring.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainingDir = './RAVIRDataset/train/training_images/*';
Files = dir(trainingDir);

for k=3:length(Files)
    fprintf("\nOpening Files: %s\n", Files(k).name);
   
    img = im2double(imread(strcat("./RAVIRDataset/train/training_images/" + Files(k).name)));
    mask = imread(strcat("./RAVIRDataset/train/training_masks/" + Files(k).name));
    
    newImg = im2uint8(normalize(img));
    saveImages(newImg, mask, Files(k).name);
   
    [newImg, newMask] = createNewImages(im2double(newImg), mask);
    normImg = zeros(size(newImg));
    
    for i=1:size(newImg,3)
       normImg(:,:,i) = im2uint8(normalize(newImg(:,:,i))); 
    end
    
    saveImages(normImg, newMask, Files(k).name);

end

function [newImg, newMask] = createNewImages(img, mask)
    numFlips = 3;
    numRotates = 5;
    numBrightness = 4; 
    numBlur = 2;

    newImg = zeros(size(img, 1), size(img, 2), numRotates + numFlips + numBrightness + numBlur);
    newMask = zeros(size(mask, 1), size(mask, 2), numRotates + numFlips + numBrightness + numBlur);

    % Image flips on each axis
    newImg(:,:,1) = flip(img, 1);
    newMask(:,:,1) = flip(mask, 1);

    newImg(:,:,2) = flip(img, 2);
    newMask(:,:,2) = flip(mask, 2);

    newImg(:,:,3) = flip(flip(img, 1), 2);
    newMask(:,:,3) = flip(flip(mask, 1), 2);
    
    % flip 1 rotates 
    newImg(:,:,4) = imrotate(newImg(:,:,1), 90, 'bicubic', 'crop');
    newMask(:,:,4) = imrotate(newMask(:,:,1), 90, 'bicubic', 'crop');
    
    % flip 2 rotates 
    newImg(:,:,5) = imrotate(newImg(:,:,2), 90, 'bicubic', 'crop');
    newMask(:,:,5) = imrotate(newMask(:,:,2), 90, 'bicubic', 'crop');

    % flip 3 rotates 
    newImg(:,:,6) = imrotate(newImg(:,:,3), 180, 'bicubic', 'crop');
    newMask(:,:,6) = imrotate(newMask(:,:,3), 180, 'bicubic', 'crop');
    newImg(:,:,7) = imrotate(newImg(:,:,3), 90, 'bicubic', 'crop');
    newMask(:,:,7) = imrotate(newMask(:,:,3), 90, 'bicubic', 'crop');
    newImg(:,:,8) = imrotate(newImg(:,:,3), 270, 'bicubic', 'crop');
    newMask(:,:,8) = imrotate(newMask(:,:,3), 270, 'bicubic', 'crop');
    
    % select 4 random images to make brightness variations of
    ind = randperm(numRotates+numFlips);
    ind = ind(1:numBrightness);
    
    % Change image brightness
    for imageIndex=1:numBrightness
        if (mod(imageIndex, 2) == 0)
            brightness = 1 + (rand() * (15) + 5) / 100.0;
            newImg(:,:,8+imageIndex) = newImg(:,:,ind(imageIndex)) .* brightness + 0.06;
        else 
            brightness = 1 - (rand() * (15) + 5) / 100.0;
            newImg(:,:,8+imageIndex) = newImg(:,:,ind(imageIndex)) .* brightness - 0.06;
        end
        
        newMask(:,:,8+imageIndex) = newMask(:,:,ind(imageIndex));
    end
    
    avgMask = ones(3,3);
    temp = im2double(newImg(:,:, 11));
    
    img = im2double(img);
    for x=1:size(img,2)
        for y=1:size(img,2)
            newImg(y,x,13) = avgNeighbors(img, avgMask, x, y);
            newImg(y,x,14) = avgNeighbors(temp, avgMask, x, y); 
        end
    end

    newMask(:,:,13) = mask;
    newMask(:,:,14) = newMask(:,:,11);
end

function normImg = normalize(img)
    img = im2double(img);
    normImg = zeros(size(img));
    hist = imhist(img);
    cdf = calcCDF(hist);
    
    lowerBound = 0.00;
    upperBound = 0.9675;
    
    lower = 999;
    upper = 0;
    
    % Calculate lower and upper bounds that correspond to percentage bounds
    for i=1:size(cdf)
       if (cdf(i) >= lowerBound && lower == 999)
           lower = i / 255.0;
       elseif (cdf(i) >= upperBound && upper == 0)
           upper = i / 255.0;
       end
    end
    
    % Apply bounds
    for y=1:size(img,1)
        for x=1:size(img,2)
            if (img(y,x) <= lower)
                normImg(y, x) = 0;
            elseif (img(y,x) >= upper)
                normImg(y, x) = 1.0;
            else 
                normImg(y, x) = (img(y, x) - lower) / upper;
            end
        end
    end
end

function cdf=calcCDF(hist)
    cdf = zeros(size(hist));
    total = sum(hist);
    
    for i=1:size(hist,1)
        for j = 1:i 
            cdf(i) = cdf(i) + hist(j);
        end
        cdf(i) = cdf(i) / total;
    end
end

% Save images
function saveImages(images, masks, name)
    if (ndims(images) == 2)
        if (ndims(masks) == 2)
            filename = name;
            imwrite(images(:,:), gray, strcat("./autoencoderDataBE/train/training_images/", filename));
            imwrite(masks(:,:), gray, strcat("./autoencoderDataBE/train/training_masks/", filename));
        else
            disp("Image and Mask should be same size"); 
        end
    else
        for file = 1:size(images, 3)
            filename = strcat(extractBetween(name, "", "."), "_", int2str(file), ".png");
            imwrite(images(:,:,file), gray, strcat("./autoencoderDataBE/train/training_images/", filename));
            imwrite(masks(:,:,file), gray, strcat("./autoencoderDataBE/train/training_masks/", filename));
        end
    end
end


