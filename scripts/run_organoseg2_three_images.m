% Batch OrganoSeg2 using the defaults initialized by the official app.
repo = 'F:\OrganoSeg2\SourceCode';
inputDir = 'F:\review\OrganoidBasic_v20211206\val';
outputDir = 'F:\review\OrganoidAgent-main\analysis-outputs\external_method_comparison_3_images\organoseg2_default';
if ~exist(outputDir, 'dir'); mkdir(outputDir); end
addpath(repo);

stems = {
 '20210727_16294_ckunspgrq2th20z824ozpde5m';
 '20210727_18955_ckw3jwvx34z2310aoet4o6hwk';
 '20210727_20499_ckw3l33aa8yqg119087nsbx46'};
bulkI = cell(size(stems));
for k = 1:numel(stems)
    img = imread(fullfile(inputDir, [stems{k} '_img.jpg']));
    if ndims(img) == 3; img = rgb2gray(img); end
    bulkI{k} = img;
end

% Defaults from OrganoSeg2.mlapp image initialization.
mainParameters.ws = 100;
mainParameters.intensityThresh = 1;
mainParameters.sizeThresh = 100;
mainParameters.checkOOF = 1;
mainParameters.checkDIC = 0;
mainParameters.checkWater = 0;
mainParameters.checkEdge = 0;
mainParameters.edgeIntensity = 0.3;
mainParameters.minimumCircularity = 0;
mainParameters.contaminantIntensity = 0;

additionalParameters.edgePreference = 'Gradient + Watershed';
additionalParameters.useReconstructedForGradient = 1;
additionalParameters.useInverted = 0;
additionalParameters.clearBorderSplit = 1;
additionalParameters.windowSizeJump = 10;
additionalParameters.imageReconstructionRadius = 5;
additionalParameters.imageClosingDegree = 3;
additionalParameters.watershedFilterSigma = 5;
additionalParameters.numberWatershedIterations = 1;

outputs = segmentFunction(numel(stems), 0, mainParameters, additionalParameters, bulkI);
for k = 1:numel(stems)
    labels = uint16(labelmatrix(outputs(k).cc));
    imwrite(labels, fullfile(outputDir, [stems{k} '_organoseg2_labels.tif']));
end
save(fullfile(outputDir, 'organoseg2_outputs.mat'), 'outputs', 'mainParameters', 'additionalParameters', '-v7.3');
exit;
