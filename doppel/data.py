import os
import swyft
import datetime


def setup_zarr_store(store_name, store_size, chunk_size, simulator):
    store = swyft.ZarrStore(file_path=store_name)
    shapes, dtypes = simulator.get_shapes_and_dtypes()
    store.init(
        N=store_size, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes
    )
    return store


def load_zarr_store(store_path):
    if os.path.exists(path=store_path):
        return swyft.ZarrStore(file_path=store_path)
    else:
        raise ValueError(f"store path ({store_path}) does not exist")


def simulate_to_store(store, simulator, batch_size, one_batch=False):
    if one_batch:
        store.simulate(
            sampler=simulator, batch_size=batch_size, max_sims=batch_size
        )
    else:
        store.simulate(sampler=simulator, batch_size=batch_size)


def setup_dataloader(
    store,
    num_workers,
    train_fraction,
    train_batch_size,
    val_batch_size,
):
    train_data = store.get_dataloader(
        num_workers=num_workers,
        batch_size=train_batch_size,
        idx_range=[0, int(train_fraction * len(store))],
        on_after_load_sample=None,
    )
    val_data = store.get_dataloader(
        num_workers=num_workers,
        batch_size=val_batch_size,
        idx_range=[
            int(train_fraction * len(store)),
            len(store) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data


def info(msg, verbose=True):
    if verbose:
        print(
            f"{datetime.datetime.now().strftime('%a %d %b %H:%M:%S')} | [doppel] | {msg}"
        )
