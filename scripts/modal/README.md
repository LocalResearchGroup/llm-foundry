# Modal

Modal is a cloud platform that enables deployment and execution of Python code in the cloud. It provides a python interface for running machine learning workloads with automatic infrastructure provisioning and GPU support.

- [Modal](https://modal.com/docs)
- Modal runs in containers. Containers are ephemeral. You can save data you want to persist on a volume or nfs mount. Volumes are higher performance than nfs. They have additional limitations that nfs does not have, but the limitations shouldn't matter for our use case. You can read the docs for more information about [Volumes](https://modal.com/docs/reference/modal.Volume) and [NFS](https://modal.com/docs/reference/modal.NetworkFileSystem). You interact with volumes from within modal functions by performing standard file operations like `open`, `read`, `write`, `close`, etc for items in the volume's mount path. You can also manage volumes and their contents using the modal CLI. After installing modal and logging in, you can interact with volumes using the `modal volume` command. Run `modal volume --help` to see the available commands and how to use them.
- When performing training runs we use aim stack for logging. You can access the aim dashboard while training by running aim and exposing it with a modal [tunnel](https://modal.com/docs/reference/modal.Tunnel). A tunnel maps a local port within the container to an external URL provided by modal that you can access over the internet.
- [Secrets](https://modal.com/docs/reference/modal.Secret) are used to store credentials and other sensitive information. You add secrets in the modal [dashboard](https://modal.com/secrets) and then access them by specifying which secrets should be available in a function run using the function decorator. The secrets are made available via environment variables from code running in the container.
- [Apps](https://modal.com/docs/reference/modal.App) are used to group modal functions together.
- [Functions](https://modal.com/docs/reference/modal.Function) are the individual units of code that are executed when you run a modal app. Functions specifically designed to run on modal have an `app.function` decorator that specifies the requirements for the function run such as GPU type, memory, timeout, volumes, secrets, etc. A local entrypoint function is required for modal python scripts to be run. It must be decorated with `@app.local_entrypoint()`.
- [Images](https://modal.com/docs/reference/modal.Image) are used to specify the base image for the container(s) that will be used to run the remote modal decorated functions.
- To run a modal script you forst


## Installation

```bash
pip install modal
```

