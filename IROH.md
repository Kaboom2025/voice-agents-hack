# iroh leader / follower

Minimal peer-to-peer data sync over [iroh](https://www.iroh.computer/) QUIC.
A `leader` accepts connections from any number of `follower` nodes and ingests
length-prefixed [`postcard`](https://docs.rs/postcard) frames containing
`EmbeddingChunk`s (per the multicam Video RAG PRD in `docs/`).

NAT traversal, relay fallback, and discovery are handled by iroh's `n0`
discovery service, so followers reach the leader from any network with just a
ticket.

## Layout

| crate      | role                                                   |
| ---------- | ------------------------------------------------------ |
| `common`   | wire types (`EmbeddingChunk`, `FollowerMsg`, `LeaderMsg`), `Ticket`, framing helpers, ALPN constants |
| `leader`   | binary: binds an iroh endpoint, prints a ticket, accepts ingest streams |
| `follower` | binary: dials a leader by ticket and streams chunks    |

ALPN: `cactus/ingest/v1`. Wire format per stream: repeated 4-byte little-endian
length + `postcard`-encoded `FollowerMsg` / `LeaderMsg`.

## Run

Two terminals (or two machines on the open internet):

```sh
# terminal A — leader
cargo run -p leader --release
# prints a base32 ticket to stdout

# terminal B — follower
# 07883c00072e3a5d03e1fc205a430c7b421c8dc8530362a1a458acee97aa0aea
cargo run -p follower --release -- <ticket> --camera-id cam-1 --interval-ms 500
```

Add as many followers as you want, each with its own `--camera-id`. The leader
logs every chunk it receives and acks them by id.

### Useful follower flags

- `--camera-id <id>`        logical id this follower reports (default `cam-0`)
- `--interval-ms <ms>`      ms between chunks (default `1000`)
- `--dim <n>`               embedding dimension to fake (default `768`)
- `--count <n>`             stop after N chunks; `0` = run forever (default `0`)

## Replacing the synthetic data

`follower/src/main.rs::make_chunk` produces random `f32` vectors as a stand-in
for the real Cactus + Gemma multimodal embedding. Wire the capture +
embedding pipeline into that function (or replace the timer loop entirely)
and the rest of the transport stays the same.

## Notes

- Tickets embed the leader's `NodeAddr` (node id + relay url + direct addrs)
  so followers can dial without any extra config.
- `MAX_FRAME_BYTES = 16 MiB` in `common/src/lib.rs` caps single-frame size to
  bound memory use against a misbehaving peer.
- For on-demand raw-clip transfer (PRD §5.1, `cactus/clip/v1`) add
  `iroh-blobs` and a second `Router::accept(...)` handler — the ingest path
  here is already content-agnostic.
