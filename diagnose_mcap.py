"""
diagnose_mcap.py — Run this on the SSH machine with cached episodes to inspect
what the MCAP messages actually contain and how they should be decoded.

Usage:
    python diagnose_mcap.py
"""
from pathlib import Path
from mcap.reader import make_reader

cache = Path("cache/place_the_bread")
episodes = sorted(cache.glob("*/episode.mcap"))
if not episodes:
    print("No cached episodes found in cache/place_the_bread/")
    exit(1)

mcap_path = episodes[0]
print(f"Inspecting: {mcap_path}\n")

with open(mcap_path, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()

    print("=" * 60)
    print("SCHEMAS")
    print("=" * 60)
    if summary and summary.schemas:
        for sid, schema in summary.schemas.items():
            print(f"\n  Schema ID {sid}:")
            print(f"    name:     {schema.name}")
            print(f"    encoding: {schema.encoding}")
            print(f"    data len: {len(schema.data)} bytes")
            try:
                text = schema.data.decode("utf-8", errors="replace")[:300]
                print(f"    data preview: {text}")
            except Exception:
                print(f"    data preview: {schema.data[:100]}")

    print("\n" + "=" * 60)
    print("CHANNELS")
    print("=" * 60)
    if summary and summary.channels:
        for cid, channel in summary.channels.items():
            print(f"  Ch {cid}: topic={channel.topic}  schema_id={channel.schema_id}  msg_encoding={channel.message_encoding}")

# Now try to decode with the appropriate decoder
print("\n" + "=" * 60)
print("ATTEMPTING DECODED MESSAGE READING")
print("=" * 60)

# Detect encoding from first channel
msg_encoding = None
if summary and summary.channels:
    first_ch = list(summary.channels.values())[0]
    msg_encoding = first_ch.message_encoding
    print(f"\nMessage encoding: {msg_encoding}")

decoder_factories = []
if msg_encoding == "protobuf":
    try:
        from mcap_protobuf.decoder import DecoderFactory
        decoder_factories = [DecoderFactory()]
        print("Using mcap-protobuf-support DecoderFactory")
    except ImportError:
        print("ERROR: pip install mcap-protobuf-support")
        exit(1)
elif msg_encoding == "json" or msg_encoding == "jsonschema":
    print("Messages are JSON encoded — will parse manually")
else:
    print(f"Unknown encoding: {msg_encoding}")

# Re-read with decoder
with open(mcap_path, "rb") as f:
    reader = make_reader(f, decoder_factories=decoder_factories)

    # Sample first 3 messages per topic
    topic_counts = {}
    for schema, channel, message, decoded in reader.iter_decoded_messages():
        topic = channel.topic
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if topic_counts[topic] > 3:
            continue

        print(f"\n--- {topic} (msg #{topic_counts[topic]}) ---")
        print(f"  Schema:    {schema.name}")
        print(f"  Raw bytes: {len(message.data)} bytes")
        print(f"  Type:      {type(decoded).__name__}")

        # Print decoded message fields
        if hasattr(decoded, "DESCRIPTOR"):
            # Protobuf message — print all fields
            for field in decoded.DESCRIPTOR.fields:
                val = getattr(decoded, field.name, None)
                if field.name == "data" and isinstance(val, bytes):
                    print(f"  .{field.name}: {len(val)} bytes (binary)")
                elif isinstance(val, (list, bytes)) and len(str(val)) > 200:
                    print(f"  .{field.name}: {type(val).__name__} len={len(val)}")
                else:
                    print(f"  .{field.name}: {val}")
        elif isinstance(decoded, dict):
            for k, v in decoded.items():
                if isinstance(v, (bytes, bytearray)) and len(v) > 100:
                    print(f"  [{k}]: {len(v)} bytes (binary)")
                elif isinstance(v, list) and len(v) > 20:
                    print(f"  [{k}]: list len={len(v)}, first 5: {v[:5]}")
                else:
                    print(f"  [{k}]: {v}")
        else:
            print(f"  Value: {str(decoded)[:300]}")

    print("\n\nDone. Total messages per topic:")
    for t, c in sorted(topic_counts.items()):
        print(f"  {t}: {c}")
