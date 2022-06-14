import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gradient_attributions(input_embeds, attention_mask, target_idx, model, x_inputs=False):
    input_embeds = input_embeds.requires_grad_(True).to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    output = model(input_embeds, attention_mask=attention_mask)[:, target_idx]
    grads = torch.autograd.grad(output, input_embeds)[0]

    if x_inputs:
        grads = grads * input_embeds

    return grads


def ig_attributions(inputs_embeds, attention_mask, target_idx, baseline, model, steps=50):
    # scale inputs and compute gradients
    interpolated_samples = __ig_interpolate_samples(baseline, inputs_embeds, steps)
    gradients = torch.tensor([])
    for sample in interpolated_samples:
        sample = sample.to(device)
        grads = gradient_attributions(sample, attention_mask, target_idx, model).to('cpu')
        gradients = torch.cat((gradients, grads), dim=0)

    gradients = (gradients[:-1] + gradients[1:]) / 2.0
    average_gradients = torch.mean(gradients, dim=0)
    integrated_gradients = (inputs_embeds - baseline) * average_gradients.to(device)
    return integrated_gradients


def __ig_interpolate_samples(baseline, target, steps):
    try:
        scaled_inputs = [(baseline + (float(i) / steps) * (target - baseline)).to('cpu') for i in range(0, steps + 1)]
    except RuntimeError:
        print()
    return scaled_inputs


def sg_attributions(inputs_embeds, attention_mask, target_idx, model, samples=50, noise_level=0.15):
    stdev = (torch.max(inputs_embeds) - torch.min(inputs_embeds)) * noise_level
    try:
        length = list(attention_mask[0]).index(0) + 1
    except ValueError:
        length = 512
    samples = __sg_generate_samples(inputs_embeds, length, stdev, samples)
    gradients = torch.tensor([]).to('cpu')
    for sample in samples:
        sample = sample.to(device)
        grads = gradient_attributions(sample, attention_mask, target_idx, model).to('cpu')
        gradients = torch.cat((gradients, grads), dim=0)

    average_gradients = torch.mean(gradients, dim=0)
    return torch.unsqueeze(average_gradients, dim=0)


def __sg_generate_samples(inputs_embeds, length, stdev, samples):
    noisy_samples = []
    padding = torch.zeros((1, 512 - length, inputs_embeds.shape[2])).to('cpu')
    for i in range(samples):
        means = torch.zeros((1, length, inputs_embeds.shape[2])).to('cpu')
        stdevs = torch.full((1, length, inputs_embeds.shape[2]), float(stdev)).to('cpu')
        noise = torch.normal(means, stdevs).to('cpu')
        noise = torch.cat((noise, padding), dim=1)
        sample = inputs_embeds.to('cpu') + noise
        noisy_samples.append(sample)

    return noisy_samples

