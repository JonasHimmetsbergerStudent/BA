
from spike_data.allen import AllenNeuropixelsWrapper

def load_wrapper(name, cache_dir, data_root):
    if name == "allen":
        from spike_data.allen import AllenNeuropixelsWrapper
        return AllenNeuropixelsWrapper(
            cache_dir=cache_dir,
            data_root=f"{data_root}/allen_neuropixels/ecephys_cache_dir")
    raise ValueError(f"Unknown dataset: {name}")


wrapper = load_wrapper(
    "allen", 
    "E:/uni/6. semester/BA Arbeit/spike-data-wrapper/data/allen", 
    "E:/uni/6. semester/BA Arbeit/spike-data-wrapper/data")

print(wrapper.get_session_ids())


