require 'lfs'
require 'paths'

require 'image'
require 'cunn'
require 'nngraph'
require 'InstanceNormalization'
torch.setdefaulttensortype('torch.FloatTensor')

local argparse = require 'argparse'
local parser = argparse()
parser:option('--input', 'Input file')
parser:option('--input_dir', 'Input directory')
parser:option('--output', 'Output file')
parser:option('--output_dir', 'Output directory')
parser:option('--model', 'Pre-trained model')
local args = parser:parse()

function predict(input_path, output_path)
    input_image = image.load(input_path)

    input_size = input_image:size()
    r = 256 / math.min(input_size[2], input_size[3])
    input_image = image.scale(input_image, r*input_size[3], r*input_size[2])
    input_size = input_image:size()
    input_image = input_image:resize(1, 3, input_size[2], input_size[3]):cuda()
    input_image:mul(2):add(-1)

    output_image = gen:forward(input_image)

    output_image:add(1):div(2)
    output_size = output_image:size()
    ouput_image = output_image:resize(3, output_size[3], output_size[4])

    image.save(output_path, output_image)
    print('Saved '..output_path)
end

if ( args['models'] == nil ) then
    print('Specify a pre-trained model')
    os.exit()
end
if ( args['input'] == nil ) and ( args['input_dir'] == nil ) then
    print('Specify an input file or input directory')
    os.exit()
end
if ( args['input_dir'] ~= nil ) and ( args['output_dir'] == nil ) then
    args['output_dir'] = 'output'
end
if ( args['output_dir'] ~= nil ) and ( not(lfs.attributes(args['output_dir'])) ) then
    paths.mkdir(args['output_dir'])
end
if ( args['output'] ~= nil ) and ( not(lfs.attributes(paths.dirname(args['output']))) ) then
    paths.mkdir(paths.dirname(args['output']))
end
if ( args['input'] ~= nil ) and ( args['output'] == nil ) then
    args['output'] = 'out_'..paths.basename(args['input'])
end

gen = torch.load(args['model'])

if not ( args['input_dir'] == nil ) then
    for file in lfs.dir(args['input_dir']) do
        if not ( string.sub(file, 1, 1) == '.' ) then
            predict(paths.concat(args['input_dir'], file), paths.concat(args['output_dir'], file))
        end
    end
else
    predict(args['input'], args['output'])
end
