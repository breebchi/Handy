//! Incremental UTF-8 framing shared by stdin and one-shot Read Aloud.

use std::io::{self, BufReader, Bytes, Read};

const MAX_PHRASE_BYTES: usize = 160;

pub(super) struct Phrases<R> {
    input: Bytes<BufReader<R>>,
    pending: String,
}

impl<R: Read> Phrases<R> {
    pub(super) fn new(input: R) -> Self {
        Self {
            input: BufReader::with_capacity(1024, input).bytes(),
            pending: String::with_capacity(MAX_PHRASE_BYTES + 4),
        }
    }

    fn read_char(&mut self) -> io::Result<Option<char>> {
        let Some(first) = self.input.next().transpose()? else {
            return Ok(None);
        };
        let width = match first {
            0..=0x7f => return Ok(Some(first.into())),
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => return Err(invalid_utf8()),
        };
        let mut bytes = [first; 4];
        for byte in &mut bytes[1..width] {
            *byte = self.input.next().transpose()?.ok_or_else(invalid_utf8)?;
        }
        let decoded = std::str::from_utf8(&bytes[..width]).map_err(|_| invalid_utf8())?;
        Ok(decoded.chars().next())
    }

    pub(super) fn next_phrase(&mut self) -> io::Result<Option<String>> {
        while let Some(ch) = self.read_char()? {
            if ch.is_whitespace() {
                if self.pending.is_empty() {
                    continue;
                }
                if ch == '\n'
                    || self
                        .pending
                        .ends_with(['.', '!', '?', ',', ';', ':', '。', '！', '？'])
                {
                    return Ok(Some(std::mem::take(&mut self.pending)));
                }
            } else if matches!(ch, '。' | '！' | '？') {
                self.pending.push(ch);
                return Ok(Some(std::mem::take(&mut self.pending)));
            }

            if self.pending.len() + ch.len_utf8() > MAX_PHRASE_BYTES {
                let split = self
                    .pending
                    .rfind(char::is_whitespace)
                    .unwrap_or(self.pending.len());
                let tail = self.pending.split_off(split);
                let phrase = std::mem::replace(&mut self.pending, tail.trim_start().to_owned());
                if !self.pending.is_empty() || !ch.is_whitespace() {
                    self.pending.push(ch);
                }
                return Ok(Some(phrase.trim_end().to_owned()));
            }
            self.pending.push(ch);
        }
        let tail = std::mem::take(&mut self.pending);
        Ok((!tail.trim().is_empty()).then_some(tail))
    }
}

fn invalid_utf8() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "Speech input must be valid UTF-8",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_short_clause_ahead_of_long_clause_and_drains_unicode_tail() {
        let long = "é".repeat(90);
        let text = format!("First clause, {long} final words");
        let mut phrases = Phrases::new(text.as_bytes());
        assert_eq!(phrases.next_phrase().unwrap().unwrap(), "First clause,");
        let mut rest = String::new();
        while let Some(phrase) = phrases.next_phrase().unwrap() {
            assert!(phrase.len() <= MAX_PHRASE_BYTES);
            rest.push_str(&phrase);
        }
        assert_eq!(rest, format!("{long} final words"));
    }

    #[test]
    fn rejects_truncated_utf8() {
        let mut phrases = Phrases::new(&[0xe2, 0x82][..]);
        assert_eq!(
            phrases.next_phrase().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
