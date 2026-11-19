pub fn decode_html_entities_into(input: &[u8], out: &mut Vec<u8>) {
  out.clear();
  let mut i = 0;
  while i < input.len() {
    if input[i] != b'&' { out.push(input[i]); i += 1; continue; }
    let rest = &input[i + 1..];
    if let Some(end) = rest.iter().take(12).position(|&b| b == b';') {
      let entity = &rest[..end];
      match entity {
        b"lt"   => { out.push(b'<'); i += end + 2; continue; }
        b"gt"   => { out.push(b'>'); i += end + 2; continue; }
        b"amp"  => { out.push(b'&'); i += end + 2; continue; }
        b"quot" => { out.push(b'"'); i += end + 2; continue; }
        b"apos" | b"#39" => { out.push(b'\''); i += end + 2; continue; }
        b"nbsp" => { out.push(b' '); i += end + 2; continue; }
        _ if entity.starts_with(b"#") => {
          let code_bytes = &entity[1..];
          let code: Option<u32> = if code_bytes.starts_with(b"x") || code_bytes.starts_with(b"X") {
            u32::from_str_radix(std::str::from_utf8(&code_bytes[1..]).unwrap_or(""), 16).ok()
          } else {
            std::str::from_utf8(code_bytes).ok().and_then(|s| s.parse().ok())
          };
          if let Some(ch) = code.and_then(char::from_u32) {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            i += end + 2;
            continue;
          }
        }
        _ => {}
      }
    }
    out.push(b'&');
    i += 1;
  }
}

// rest가 "[["로 시작할 때 짝이 맞는 "]]"의 위치 (중첩 [[ ]] 고려)
fn matching_link_end(rest: &[u8]) -> Option<usize> {
  let mut depth = 0usize;
  let mut j = 0;
  while j + 1 < rest.len() {
    if rest[j] == b'[' && rest[j + 1] == b'[' { depth += 1; j += 2; }
    else if rest[j] == b']' && rest[j + 1] == b']' {
      depth -= 1;
      if depth == 0 { return Some(j); }
      j += 2;
    } else { j += 1; }
  }
  None
}

pub fn strip_wiki_markup_into(input: &[u8], out: &mut Vec<u8>) {
  out.clear();
  let mut i = 0;
  let mut tmpl_depth = 0i32;
  let mut table_depth = 0i32;

  while i < input.len() {
    let rest = &input[i..];

    // 템플릿을 표보다 먼저 처리: 표 안의 {{x|}} 에 들어 있는 "|}"를 표 끝으로 오인하지 않도록
    if rest.starts_with(b"{{") { tmpl_depth += 1; i += 2; continue; }
    if tmpl_depth > 0 {
      if rest.starts_with(b"}}") { tmpl_depth -= 1; i += 2; }
      else { i += 1; }
      continue;
    }

    // 위키 표 {| ... |} 는 셀 구분자·속성이 대부분이라 통째로 제외 (중첩 표 포함)
    if rest.starts_with(b"{|") { table_depth += 1; i += 2; continue; }
    if table_depth > 0 {
      if rest.starts_with(b"|}") { table_depth -= 1; i += 2; }
      else { i += 1; }
      continue;
    }

    // 굵게/기울임 ('' ''' ''''') 은 지워서 앞뒤 글자가 자연스럽게 이어지게 함
    if rest.starts_with(b"''") {
      i += rest.iter().take_while(|&&b| b == b'\'').count();
      continue;
    }

    if rest.starts_with(b"<!--") {
      i += rest.windows(3).position(|w| w == b"-->").map_or(rest.len(), |p| p + 3);
      continue;
    }

    // <ref>, <ref name=...>, <ref .../> 만 처리 (<references> 등은 아래 일반 태그 처리)
    if rest.starts_with(b"<ref") && rest.get(4).is_some_and(|&b| b == b'>' || b == b'/' || b.is_ascii_whitespace()) {
      let Some(gt) = rest.iter().position(|&b| b == b'>') else { i += 4; continue; };
      if rest[gt - 1] == b'/' {
        i += gt + 1;
      } else if let Some(p) = rest[gt + 1..].windows(6).position(|w| w == b"</ref>") {
        i += gt + 1 + p + 6;
      } else {
        i += gt + 1;
      }
      continue;
    }

    if rest.starts_with(b"<") {
      i += rest.iter().position(|&b| b == b'>').map_or(1, |p| p + 1);
      continue;
    }

    if rest.starts_with(b"[[") {
      if let Some(end) = matching_link_end(rest) {
        let content = &rest[2..end];
        if content.iter().take(40).any(|&b| b == b':') {
          // namespace link (File:, Category:, etc.) — 캡션 안의 중첩 링크까지 통째로 제외
        } else {
          let shown = content.iter().position(|&b| b == b'|').map_or(content, |pipe| &content[pipe + 1..]);
          // 표시 텍스트 안에 다시 마크업(중첩 링크, 굵게 등)이 있을 때만 재귀 처리
          if shown.iter().any(|&b| matches!(b, b'[' | b'\'' | b'{' | b'<')) {
            let mut inner = Vec::new();
            strip_wiki_markup_into(shown, &mut inner);
            out.extend_from_slice(&inner);
          } else {
            out.extend_from_slice(shown);
          }
        }
        i += end + 2;
      } else {
        i += 2;
      }
      continue;
    }

    if rest.starts_with(b"[http") || rest.starts_with(b"[//") {
      i += rest.iter().position(|&b| b == b']').map_or(1, |p| p + 1);
      continue;
    }

    out.push(input[i]);
    i += 1;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn strip(s: &str) -> String {
    let mut out = Vec::new();
    strip_wiki_markup_into(s.as_bytes(), &mut out);
    String::from_utf8(out).unwrap()
  }

  #[test]
  fn ref_tags_are_removed_without_eating_text() {
    // 뒤쪽 self-closing ref 때문에 사이 본문이 사라지면 안 됨
    assert_eq!(strip("aa<ref>cite</ref>bb cc<ref name=\"x\"/>dd"), "aabb ccdd");
    assert_eq!(strip("x<ref name=a>c</ref>y"), "xy");
    assert_eq!(strip("p<references/>q"), "pq");
  }

  #[test]
  fn templates_comments_and_links() {
    assert_eq!(strip("a{{cite|x={{y}}}}b"), "ab");
    assert_eq!(strip("a<!-- hidden -->b"), "ab");
    assert_eq!(strip("[[Target|shown]] [[plain]] [[File:x.png|cap]]"), "shown plain ");
    assert_eq!(strip("see [http://x.org label] now"), "see  now");
  }

  #[test]
  fn tables_are_removed() {
    assert_eq!(strip("a\n{| class=wikitable\n|-\n| x || y\n|}\nb"), "a\n\nb");
    // 중첩 표, 그리고 표 안 템플릿의 "|}}" 를 표 끝으로 오인하지 않아야 함
    assert_eq!(strip("a{| {{t|}} {| inner |} still |}b"), "ab");
  }

  #[test]
  fn nested_links_and_bold() {
    assert_eq!(strip("x [[File:a.png|thumb|cap with [[link]] in it]] y"), "x  y");
    assert_eq!(strip("[[a|b [[c|e]] d]]"), "b e d");
    assert_eq!(strip("[[Paris|'''Paris''']] is"), "Paris is");
    assert_eq!(strip("'''Paris''' is ''very'' old"), "Paris is very old");
    assert_eq!(strip("don't"), "don't");
  }

  #[test]
  fn html_entities() {
    let mut out = Vec::new();
    decode_html_entities_into(b"a&lt;b&gt;&amp;&#65;&#x42;&nbsp;&bogus;", &mut out);
    assert_eq!(out, b"a<b>&AB &bogus;");
  }
}
