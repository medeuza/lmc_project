from .matching import weight_matching


def hub_merge(params_list, n_layers):
    hub = params_list[0]
    aligned = [hub]

    for params in params_list[1:]:
        _, aligned_params = weight_matching(hub, params, n_layers)
        aligned.append(aligned_params)

    return {
        key: sum(params[key] for params in aligned) / len(aligned)
        for key in hub
    }